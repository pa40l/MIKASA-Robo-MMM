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
