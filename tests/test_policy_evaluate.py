import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.policies import evaluate as ev

REPO = Path(__file__).resolve().parents[1]


def chunk(values, gripper=None):
    actions = np.zeros((len(values), 13), np.float32)
    actions[:, 0] = list(values)
    if gripper is not None:
        actions[:, ev.GRIPPER] = gripper
    return actions


def test_midpoint_averages_with_the_next_target_and_holds_the_gripper():
    steps = ev.control_steps(chunk([0, 2, 4, 6], gripper=[-1, -1, 1, 1]), 2, "midpoint")
    assert steps[:, 0].tolist() == [0, 1, 2, 3]  # the last executed target averages with the chunk's next
    assert steps[:, ev.GRIPPER].tolist() == [-1, -1, -1, -1]


def test_midpoint_holds_the_last_target_without_a_next_one():
    assert ev.control_steps(chunk([0, 2]), 16, "midpoint")[:, 0].tolist() == [0, 1, 2, 2]


def test_hold_repeats_each_target():
    assert ev.control_steps(chunk([0, 2, 4]), 2, "hold")[:, 0].tolist() == [0, 0, 2, 2]


def test_limit_is_twice_the_longest_episode_in_control_steps():
    assert ev.episode_limit([500, 690, 10]) == 2760


def test_seed_files(tmp_path):
    for value in ([3, 1], {"episode_seeds": [3, 1]}, {"seeds": [3, 1]}):
        path = tmp_path / "seeds.json"
        path.write_text(json.dumps(value))
        assert ev.read_seeds(path) == [3, 1]
    path.write_text("[1, 1]")
    with pytest.raises(ValueError):
        ev.read_seeds(path)


class FakeEnv:
    def __init__(self, success_from=None, success_only_at=None, fail_at=None):
        self.success_from, self.success_only_at, self.fail_at = success_from, success_only_at, fail_at
        self.actions = []

    def step(self, action):
        self.actions.append(np.array(action))
        n = len(self.actions)
        success = (self.success_from is not None and n >= self.success_from) or n == self.success_only_at
        terminated = success or (self.fail_at is not None and n >= self.fail_at)
        return {"step": n}, 0.0, terminated, False, {"success": success}


class FakeClient:
    def __init__(self, actions):
        self.actions, self.ingested, self.queries, self.frames = actions, [], [], 0

    @staticmethod
    def observation(obs, prompt):
        return {"step": obs["step"], "prompt": prompt}

    def ingest(self, observation):
        self.ingested.append(observation["step"])
        self.frames += 1

    def infer(self, observation):
        self.queries.append(observation["step"])
        return self.actions


def run(env, client, **kwargs):
    options = dict(max_steps=100, horizon=2, execution="midpoint")
    options.update(kwargs)
    return ev.run_episode(env, {"step": 0}, client, "go", **options)


def test_episode_stops_at_success_and_the_policy_sees_every_second_step():
    env, client = FakeEnv(success_from=7), FakeClient(chunk(range(20)))
    out = run(env, client)
    assert (out["status"], out["success"], out["simulator_steps"]) == ("success", True, 7)
    assert client.queries == [0, 4] and client.ingested == [0, 2, 4, 6]
    assert [float(a[0]) for a in env.actions[:4]] == [0, 0.5, 1, 1.5]


def test_episode_fails_when_the_task_terminates_without_success():
    out = run(FakeEnv(fail_at=3), FakeClient(chunk(range(20))))
    assert (out["status"], out["success"], out["simulator_steps"]) == ("fail", False, 3)


def test_episode_times_out_at_the_limit():
    out = run(FakeEnv(), FakeClient(chunk(range(20))), max_steps=10)
    assert (out["status"], out["success"], out["simulator_steps"]) == ("timeout", False, 10)


def recorded(monkeypatch, actions, horizon=4):
    monkeypatch.setattr(ev.RecordedClient, "observation", staticmethod(FakeClient.observation))
    client = ev.RecordedClient(horizon)
    client.load(actions)
    client.reset("session")
    return client


def test_recorded_episode_plays_to_the_end_and_is_judged_at_the_last_frame(monkeypatch):
    env = FakeEnv(success_from=5)
    out = run(env, recorded(monkeypatch, chunk(range(10))), horizon=4, stop_on_success=False)
    assert (out["status"], out["success"], out["simulator_steps"], out["queries"]) == ("replayed", True, 20, 3)
    assert float(env.actions[7][0]) == 3.5  # a query's last target averages with the next query's first
    assert float(env.actions[-1][0]) == 9  # the episode's last target is held
    out = run(FakeEnv(success_only_at=5), recorded(monkeypatch, chunk(range(10))), horizon=4, stop_on_success=False)
    assert (out["success"], out["success_once"]) == (False, True)


def test_seeds_of_training_episodes_are_refused(tmp_path):
    source = SimpleNamespace(lengths=[10, 20], env_id="MikasaSameDrawer-v0", instruction="open",
                             training_seeds={5, 6})
    hf = tmp_path / "hf"  # the HF layout: <task>/validation_seeds.json
    (hf / "samedrawer").mkdir(parents=True)
    (hf / "samedrawer/validation_seeds.json").write_text(json.dumps({"rule": "first 100", "seeds": [5, 7, 8]}))
    args = SimpleNamespace(seeds_dir=hf, seeds=[], episodes=2, start_seed=0, max_steps=None,
                           backend="memer", shard="0/1")
    with pytest.raises(ValueError, match="training"):
        ev.plan_rollouts(args, {"samedrawer": source})
    override = tmp_path / "seeds.json"  # --seeds overrides the folder's file
    override.write_text("[7, 8, 9]")
    args.seeds = [f"samedrawer={override}"]
    plan, settings = ev.plan_rollouts(args, {"samedrawer": source})
    assert [seed for _, seed, _ in plan] == [7, 8] and settings["samedrawer"]["max_steps"] == 80
    args.shard = "1/2"
    assert [seed for _, seed, _ in ev.plan_rollouts(args, {"samedrawer": source})[0]] == [8]


def test_tasks_take_turns_so_every_shard_covers_all_of_them():
    sources = {task: SimpleNamespace(lengths=[10], env_id=task, instruction=task, training_seeds=set())
               for task in ("samedrawer", "veggie")}
    args = SimpleNamespace(seeds_dir=None, seeds=[], episodes=3, start_seed=0, max_steps=None,
                           backend="memer", shard="0/1")
    plan, settings = ev.plan_rollouts(args, sources)
    assert [(task, seed) for task, seed, _ in plan] == [
        ("samedrawer", 0), ("veggie", 0), ("samedrawer", 1), ("veggie", 1), ("samedrawer", 2), ("veggie", 2)]
    assert settings["veggie"]["episodes"] == 3
    args.episodes = 8
    shards = [[(task, seed) for task, seed, _ in ev.plan_rollouts(SimpleNamespace(**{**vars(args), "shard": f"{k}/4"}),
                                                                  sources)[0]] for k in range(4)]
    assert shards[0] == [("samedrawer", 0), ("veggie", 3), ("samedrawer", 4), ("veggie", 7)]
    assert sorted(item for shard in shards for item in shard) == sorted(
        (task, seed) for task in sources for seed in range(8))  # every rollout once


def load_server():
    spec = importlib.util.spec_from_file_location("serve_high", REPO / "vla/memer/serve_high.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation(prompt="put the cup back"):
    return {"observation.images.fetch_hand": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            "observation.state": np.zeros(12, np.float32), "prompt": prompt}


def test_memer_client_sends_the_frames_since_its_last_query_and_prompts_the_subtask():
    calls, prompts = [], []
    replies = iter([{"timestep": 0, "parse_ok": False, "current_subtask": None, "memory_indices_after": []},
                    {"timestep": 2, "parse_ok": True, "current_subtask": "open the drawer",
                     "memory_indices_after": [1]}])

    def call(command, **payload):
        calls.append((command, payload))
        return {"ok": True} if command == "reset" else next(replies)

    def infer(request):
        prompts.append(request["prompt"])
        return {"actions": np.zeros((20, 13), np.float32)}

    client = object.__new__(ev.MemerClient)
    client.call, client.low = call, SimpleNamespace(infer=infer)
    client.reset("session")
    client.ingest(observation())
    client.infer(observation())
    client.ingest(observation())
    client.ingest(observation())
    client.infer(observation())
    assert calls[0] == ("reset", {"instruction": "put the cup back"})
    assert [len(payload["frames"]) for command, payload in calls if command == "step"] == [1, 2]
    assert prompts == ["put the cup back", "open the drawer"]  # no subtask yet: the instruction
    assert client.extra() == {"subtasks": [{"frame": 2, "subtask": "open the drawer"}], "high_calls": 2,
                              "parse_failures": 1, "instruction_prompts": 1, "memory_frames": [1]}


def test_high_level_server_decodes_the_frames_and_reports_errors():
    server = load_server()

    @dataclasses.dataclass
    class Result:
        frames: int

    class Policy:
        def reset(self, instruction):
            self.instruction = instruction

        def step(self, *frames):
            self.frames = frames
            return Result(len(frames))

    policy = Policy()
    assert server.handle(policy, {"cmd": "reset", "instruction": "go"}, {}) == {"ok": True}
    frame = ev.encode_frame(observation())
    assert list(frame) == ["observation.images.fetch_hand"]
    assert server.handle(policy, {"cmd": "step", "frames": [frame, frame]}, {}) == {"frames": 2}
    np.testing.assert_array_equal(policy.frames[1]["observation.images.fetch_hand"],
                                  observation()["observation.images.fetch_hand"])

    class Connection:
        def __init__(self, requests):
            self.requests, self.sent, self.closed = list(requests), [], False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.closed = True

        def recv(self):
            if not self.requests:
                raise EOFError
            return self.requests.pop(0)

        def send(self, reply):
            self.sent.append(reply)

    connection = Connection([{"cmd": "fly"}, {"cmd": "metadata"}])
    server.serve(connection, policy, {"model": "m"})
    assert "unknown command" in connection.sent[0]["error"] and connection.sent[1] == {"model": "m"}
    assert connection.closed


def test_high_level_server_gives_every_client_its_own_memory_and_one_model():
    import threading
    from multiprocessing import Pipe

    server = load_server()
    running, peak, calls = [0], [0], []

    @dataclasses.dataclass
    class Result:
        prompt: str

    class Model:  # records how many predictions overlap
        def predict(self, prompt, images):
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            calls.append(prompt)
            threading.Event().wait(0.01)
            running[0] -= 1
            return Result(prompt)

    class Policy:  # MemERDeploymentPolicy's shape: a client's own frames, the shared model
        def __init__(self, model):
            self.model, self.frames = model, 0

        def reset(self, instruction):
            self.instruction, self.frames = instruction, 0

        def step(self, *frames):
            self.frames += len(frames)
            return self.model.predict(f"{self.instruction}:{self.frames}", frames)

    shared = server.SerialPredictor(Model())
    clients, threads = [], []
    for name in ("a", "b", "c"):
        near, far = Pipe()
        threads.append(threading.Thread(target=server.serve, args=(far, Policy(shared), {}), daemon=True))
        threads[-1].start()
        clients.append((name, near))
    frame = ev.encode_frame(observation())
    for name, near in clients:
        near.send({"cmd": "reset", "instruction": name})
    assert [near.recv() for _, near in clients] == [{"ok": True}] * 3
    for frames in (1, 2):
        for _, near in clients:
            near.send({"cmd": "step", "frames": [frame] * frames})
        expected = [{"prompt": f"{name}:{1 if frames == 1 else 3}"} for name, _ in clients]
        assert [near.recv() for _, near in clients] == expected
    for _, near in clients:
        near.close()
    for thread in threads:
        thread.join(2)
    assert not any(thread.is_alive() for thread in threads)  # a hung-up client ends its thread
    assert peak[0] == 1 and len(calls) == 6  # one prediction at a time


def test_done_elsewhere_reads_the_same_runs_other_shard_layouts(tmp_path):
    header = {"backend": "memer", "execution": "midpoint", "shard": "0/8", "tasks": {"veggie": {"episodes": 2}}}
    def write(name, shard, rollouts, **change):
        (tmp_path / name).write_text(json.dumps({**header, **change, "shard": shard, "rollouts": rollouts}))
    write("shard-0-of-4.json", "0/4", [{"task": "veggie", "seed": 1}])
    write("other-settings.json", "1/4", [{"task": "veggie", "seed": 2}], execution="hold")
    write("shard-0-of-8.json", "0/8", [{"task": "veggie", "seed": 3}])  # this shard's own output
    (tmp_path / "broken.json").write_text("{")
    assert ev.done_elsewhere(tmp_path, tmp_path / "shard-0-of-8.json", header) == {("veggie", 1)}


def test_observation_leaves_out_a_scene_camera_and_needs_the_three_of_the_robot():
    def sensors(*names):
        shape = {"fetch_hand": (1, 128, 128, 3)}
        return {name: {"rgb": np.zeros(shape.get(name, (1, 256, 256, 3)), np.uint8)} for name in names}

    obs = {"agent": {"qpos": np.zeros((1, 15), np.float32)},
           "sensor_data": sensors(*ev.CAMERAS, "base_camera")}  # jezv's scenes add their own camera
    observation = ev.mikasa_observation(obs, "go")
    assert sorted(key for key in observation if key.startswith("observation.images.")) == sorted(
        f"observation.images.{camera}" for camera in ev.CAMERAS)
    with pytest.raises(ValueError, match="lacks"):
        ev.mikasa_observation({**obs, "sensor_data": sensors("fetch_hand", "base_camera")}, "go")


# MemoryVLA: training actions are the recorded targets minus the frame's own joints for the arm,
# head and torso (vla/memoryvla/patches/0001-fetch-datasets.patch, fetch_dataset_transform)
FETCH_ARM_STATE, FETCH_HEAD_STATE, FETCH_TORSO_STATE = [2, 4, 5, 6, 7, 8, 9], [1, 3], [0]


def training_delta(action, state):
    return np.concatenate([action[:7] - state[FETCH_ARM_STATE], action[7:8], action[8:10] - state[FETCH_HEAD_STATE],
                           action[10:11] - state[FETCH_TORSO_STATE], action[11:13]])


def robot(state):
    return {"agent": {"qpos": np.concatenate([[9.0, 9.0, 9.0], state]).astype(np.float32)[None]}}


def test_memoryvla_targets_undo_the_training_transform():
    rng = np.random.default_rng(0)
    state, action = rng.normal(size=12).astype(np.float32), rng.normal(size=13).astype(np.float32)
    delta = training_delta(action, state)
    np.testing.assert_allclose(ev.MemoryVLAClient.targets(delta[None], robot(state))[0], action, atol=1e-6)


class JointEnv(FakeEnv):
    """A robot whose first joint state channel (torso, action 10) counts the control steps."""

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        state = np.zeros(12, np.float32)
        state[0] = len(self.actions)
        return robot(state), reward, terminated, truncated, info


class RelativeClient(FakeClient):
    targets = staticmethod(ev.MemoryVLAClient.targets)

    @staticmethod
    def observation(obs, prompt):
        return {"step": int(obs["agent"]["qpos"][0, 3]), "prompt": prompt}


@pytest.mark.parametrize("execution", ["hold", "midpoint"])
def test_relative_chunks_are_made_absolute_from_the_joints_of_each_frame(execution):
    deltas = np.zeros((4, 13), np.float32)
    deltas[:, 10] = 0.5  # torso: half a unit above wherever it is
    env, client = JointEnv(), RelativeClient(deltas)
    out = ev.run_episode(env, robot(np.zeros(12, np.float32)), client, "go", max_steps=8, horizon=4,
                         execution=execution)
    assert out["simulator_steps"] == 8 and client.queries == [0]
    # each 10 Hz target starts from the joints measured after the previous pair: 0, 2, 4, 6
    assert [float(a[10]) for a in env.actions] == [0.5, 0.5, 2.5, 2.5, 4.5, 4.5, 6.5, 6.5]


def test_memoryvla_client_resets_with_the_task_statistics_and_sends_the_frame_index():
    calls = []

    def call(command, **payload):
        calls.append((command, payload))
        return {"ok": True} if command == "reset" else {"actions": np.zeros((16, 13)).tolist(), "memory": 1}

    client = object.__new__(ev.MemoryVLAClient)
    client.call = call
    with pytest.raises(ValueError):
        client.reset("session")
    client.reset("session", task="veggie")
    client.ingest(observation())
    assert client.infer(observation()).shape == (16, 13)
    for _ in range(16):
        client.ingest(observation())
    client.infer(observation())
    assert calls[0] == ("reset", {"instruction": "put the cup back", "unnorm_key": "fetch_veggie"})
    assert [payload["frame"] for command, payload in calls if command == "step"] == [0, 16]
    assert client.extra() == {"memory_entries": 1}


def load_memoryvla_server():
    import sys

    spec = importlib.util.spec_from_file_location("memoryvla_serve", REPO / "vla/memoryvla/serve.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass resolves its annotations through the module
    spec.loader.exec_module(module)
    return module


class FakeVLA:
    """predict_action's memory handling (vla/memory_vla.py): reset on the first frame, stamp, append."""

    def __init__(self):
        self.cog_mem_bank, self.per_mem_bank = SimpleNamespace(bank={}), SimpleNamespace(bank={})
        self.cur_timestep, self.norm_stats, self.seen = 0, {"fetch_veggie": {}, "fetch_samedrawer": {}}, []

    def predict_action(self, *, image, image_wrist, image_wrist_2, instruction, unnorm_key, episode_first_frame,
                       **kwargs):
        if episode_first_frame == "True":
            self.cog_mem_bank.bank, self.per_mem_bank.bank, self.cur_timestep = {}, {}, 0
        assert image.size == image_wrist.size == image_wrist_2.size == (224, 224)
        self.seen.append((instruction, [t for t, _ in self.cog_mem_bank.bank.get(0, [])], self.cur_timestep))
        for bank in (self.cog_mem_bank.bank, self.per_mem_bank.bank):
            bank.setdefault(0, []).append((self.cur_timestep, instruction))
        self.cur_timestep += 1
        return np.full((16, 13), len(self.seen), np.float32), None


def memoryvla_frame():
    return ev.encode_frame({"observation.images.left_base_camera_link": np.zeros((256, 256, 3), np.uint8),
                            "observation.images.right_base_camera_link": np.zeros((256, 256, 3), np.uint8),
                            "observation.images.fetch_hand": np.zeros((128, 128, 3), np.uint8)})


@pytest.mark.parametrize("timestep", ["call", "frame"])
def test_memoryvla_server_keeps_every_session_its_own_memory(timestep):
    server = load_memoryvla_server()
    vla = FakeVLA()
    model = server.SessionModel(vla, timestep=timestep)
    a, b = server.Session(), server.Session()
    frame = server.decode_frame(memoryvla_frame())
    model.reset(a, "a", "fetch_veggie")
    model.reset(b, "b", "fetch_samedrawer")
    with pytest.raises(KeyError):
        model.reset(server.Session(), "c", "fetch_unknown")
    with pytest.raises(ValueError):
        model.step(server.Session(instruction="d", unnorm_key="fetch_veggie"), 16, frame)  # not at frame 0
    model.step(a, 0, frame)
    model.step(b, 0, frame)
    model.step(a, 16, frame)
    reply = model.step(a, 32, frame)
    model.step(b, 16, frame)
    later = 16 if timestep == "frame" else 1
    assert vla.seen == [("a", [], 0), ("b", [], 0), ("a", [0], later), ("a", [0, later], 2 * later),
                        ("b", [0], later)]
    assert reply["calls"] == 3 and reply["memory"] == 3 and len(reply["actions"]) == 16
    assert vla.cog_mem_bank.bank == {}  # nothing of a session stays in the model between steps


def test_memoryvla_server_replies_errors_and_serves_until_the_client_hangs_up():
    import threading
    from multiprocessing import Pipe

    server = load_memoryvla_server()
    model = server.SessionModel(FakeVLA())
    near, far = Pipe()
    thread = threading.Thread(target=server.serve, args=(far, model, {"step": 100000}), daemon=True)
    thread.start()
    near.send({"cmd": "metadata"})
    assert near.recv() == {"step": 100000}
    near.send({"cmd": "step", "frame": 0, "images": memoryvla_frame()})
    assert "step before reset" in near.recv()["error"]
    near.send({"cmd": "reset", "instruction": "go", "unnorm_key": "fetch_veggie"})
    assert near.recv() == {"ok": True}
    near.send({"cmd": "step", "frame": 0, "images": memoryvla_frame()})
    assert near.recv()["calls"] == 1
    near.close()
    thread.join(2)
    assert not thread.is_alive()
