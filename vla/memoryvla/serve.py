"""MemoryVLA as a local server for closed-loop evaluation (utils/policies/evaluate.py --backend memoryvla).

Runs in the MemoryVLA env; the authors' code comes from third_party/MemoryVLA:
    ~/envs/memvla/bin/python vla/memoryvla/serve.py --checkpoint <run>/checkpoints/step-....pt \
        [--port 8200] [--timestep call|frame]

The checkpoint's run folder must hold config.json, config.yaml and dataset_statistics.json, as
load_vla and the authors' deploy.py expect.

Many clients, one model: every connection (an evaluation shard) gets a thread and an episode
memory of its own. The model keeps one episode's memory banks in itself, so each step swaps the
session's banks and time step in, runs, and takes them back out, one step at a time.

Inference as the authors' MIKASA evaluation (evaluation/mikasa/eval_mikasa_with_deploy.py): bf16,
CFG 1.5, 10 DDIM steps, every image cropped to a square, resized to 224 (LANCZOS) and center-cropped
to 90 % of its area (training crops 90 % at random), the memory reset on the episode's first call.
Our three cameras fill MemoryVLA's three-view mode as in training (patches/0001-fetch-datasets):
the left base camera as the main image, fetch_hand as the first wrist image, the right base camera
as the second.

The memory time stamp: the authors' inference counts calls (predict_action's cur_timestep), while
training with the group loader stamps every memory entry with its 10 Hz frame index.
--timestep call (default) keeps the authors' count; frame uses the client's frame index.

A step replies with the whole chunk unnormalized, in the training action space: arm, head and
torso as deltas from each frame's own joints, the gripper absolute, the base velocities as they
are (patches/0001 fetch_dataset_transform). The client adds the measured joints back at every
executed frame. Connections are local and need the key in $MEMORYVLA_AUTHKEY (default "memoryvla").
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import re
import sys
import threading
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MEMVLA = REPO / "third_party/MemoryVLA"
AUTHKEY_ENV = "MEMORYVLA_AUTHKEY"  # utils/policies/evaluate.py reads the same variable
VIEWS = ("observation.images.left_base_camera_link",   # image (primary)
         "observation.images.fetch_hand",              # image_wrist (left_wrist slot)
         "observation.images.right_base_camera_link")  # image_wrist_2 (right_wrist slot)
NOT_MODEL_ARGS = ("model_id_or_path", "saved_model_path", "pretrained_checkpoint")


def decode_frame(frame):
    """utils/policies/evaluate.py encode_frame, reversed: camera key -> HxWx3 uint8."""
    return {key: np.frombuffer(data, np.uint8).reshape(shape) for key, (data, shape) in frame.items()}


def policy_image(array, size=(224, 224), scale=0.9):
    """The authors' resize_image_for_policy: square crop, LANCZOS resize, center crop of `scale` area."""
    from PIL import Image

    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    w, h = image.size
    if w < h:
        raise ValueError(f"expected a square or wide image, got {w}x{h}")
    left = (w - h) // 2
    image = image.crop((left, 0, left + h, h)).resize(size, resample=Image.LANCZOS)
    w, h = image.size
    new_w, new_h = int(w * math.sqrt(scale)), int(h * math.sqrt(scale))
    margin_w, margin_h = (w - new_w) // 2, (h - new_h) // 2
    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    return image.resize(size, resample=Image.LANCZOS)


@dataclasses.dataclass
class Session:
    """One client's episode: its instruction, dataset statistics and memory."""
    instruction: str | None = None
    unnorm_key: str | None = None
    cog: list = dataclasses.field(default_factory=list)
    per: list = dataclasses.field(default_factory=list)
    calls: int = 0


class SessionModel:
    """One loaded MemoryVLA for every client, one prediction at a time.

    `vla` needs predict_action and the attributes it keeps an episode in: cog_mem_bank.bank,
    per_mem_bank.bank (both {episode id: entries}; predict_action uses episode 0) and cur_timestep.
    """

    def __init__(self, vla, *, timestep="call", cfg_scale=1.5, ddim_steps=10):
        if timestep not in ("call", "frame"):
            raise ValueError(f"timestep must be call or frame, not {timestep!r}")
        self.vla, self.timestep, self.cfg_scale, self.ddim_steps = vla, timestep, cfg_scale, ddim_steps
        self.lock = threading.Lock()

    def reset(self, session, instruction, unnorm_key):
        stats = getattr(self.vla, "norm_stats", None)
        if stats is not None and unnorm_key not in stats:
            raise KeyError(f"no dataset statistics {unnorm_key!r}; the model has {sorted(stats)}")
        session.instruction, session.unnorm_key = instruction, unnorm_key
        session.cog, session.per, session.calls = [], [], 0

    def step(self, session, frame, observation):
        if session.instruction is None:
            raise RuntimeError("step before reset")
        if session.calls == 0 and frame != 0:
            raise ValueError(f"an episode's first query must be at frame 0, not {frame}")
        images = [policy_image(observation[view]) for view in VIEWS]
        with self.lock:
            vla = self.vla
            vla.cog_mem_bank.bank = {0: session.cog}
            vla.per_mem_bank.bank = {0: session.per}
            vla.cur_timestep = frame if self.timestep == "frame" else session.calls
            actions, _ = vla.predict_action(
                image=images[0], image_wrist=images[1], image_wrist_2=images[2],
                instruction=session.instruction, unnorm_key=session.unnorm_key,
                cfg_scale=self.cfg_scale, use_ddim=True, num_ddim_steps=self.ddim_steps,
                episode_first_frame="True" if session.calls == 0 else "False")
            session.cog = vla.cog_mem_bank.bank.get(0, [])
            session.per = vla.per_mem_bank.bank.get(0, [])
            vla.cog_mem_bank.bank, vla.per_mem_bank.bank = {}, {}
        session.calls += 1
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise FloatingPointError(f"bad action chunk {actions.shape}")
        return {"actions": actions.tolist(), "calls": session.calls, "memory": len(session.cog)}


def handle(model, session, request, metadata):
    command = request.get("cmd")
    if command == "metadata":
        return metadata
    if command == "reset":
        model.reset(session, request["instruction"], request["unnorm_key"])
        return {"ok": True}
    if command == "step":
        return model.step(session, int(request["frame"]), decode_frame(request["images"]))
    raise ValueError(f"unknown command {command!r}")


def serve(connection, model, metadata):
    """One client's requests until it hangs up; then the connection is closed."""
    session = Session()
    with connection:
        while True:
            try:
                request = connection.recv()
            except (EOFError, OSError):
                return
            try:
                reply = handle(model, session, request, metadata)
            except Exception as error:  # the client raises it; the server stays up for the others
                logging.exception("request %r failed", request.get("cmd"))
                reply = {"error": f"{type(error).__name__}: {error}"}
            connection.send(reply)


def load(checkpoint):
    """load_vla with the run's config.yaml, as the authors' deploy.py; bf16 on the GPU."""
    import torch
    import yaml

    sys.path.insert(0, str(MEMVLA))
    from vla import load_vla

    run = Path(checkpoint).resolve().parents[1]
    config = yaml.safe_load((run / "config.yaml").read_text()) or {}
    kwargs = {key: value for key, value in config.items() if key not in NOT_MODEL_ARGS}
    vla = load_vla(model_id_or_path=str(checkpoint), load_for_training=False, **kwargs)
    vla = vla.to(torch.device("cuda")).to(torch.bfloat16).eval()
    torch.cuda.empty_cache()
    return vla, config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="<run>/checkpoints/step-*.pt")
    parser.add_argument("--timestep", choices=("call", "frame"), default="call")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    vla, config = load(args.checkpoint)
    model = SessionModel(vla, timestep=args.timestep, cfg_scale=args.cfg_scale, ddim_steps=args.ddim_steps)
    step = re.match(r"step-0*(\d+)-", args.checkpoint.name)
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "step": int(step.group(1)) if step else None,
                "timestep": args.timestep, "cfg_scale": args.cfg_scale, "ddim_steps": args.ddim_steps,
                "chunk": int(config.get("future_action_window_size", 15)) + 1,
                "action_dim": config.get("action_dim"), "mem_length": config.get("mem_length"),
                "unnorm_keys": sorted(getattr(vla, "norm_stats", {}) or {}), "views": list(VIEWS)}
    authkey = os.environ.get(AUTHKEY_ENV, "memoryvla").encode()
    with Listener((args.host, args.port), authkey=authkey) as listener:
        logging.info("Serving MemoryVLA %s (timestep %s) on %s:%d", args.checkpoint.name, args.timestep,
                     args.host, args.port)
        while True:
            try:
                connection = listener.accept()
            except AuthenticationError:
                logging.warning("refused a client with another key")
                continue
            except (EOFError, OSError) as error:  # a client gone before the handshake
                logging.warning("dropped a connection before the handshake: %r", error)
                continue
            logging.info("client %s connected", listener.last_accepted)
            threading.Thread(target=serve, args=(connection, model, metadata), daemon=True).start()


if __name__ == "__main__":
    main()
