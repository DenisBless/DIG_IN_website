"""Render the recorded DMC episodes of the project page into assets/rl/<Task>.mp4,
with a poster image (assets/rl/<Task>.jpg) and their returns (assets/rl/episodes.json).

An episode is a greedy rollout of a trained DIG-IN policy, the best of the 64 greedy
episodes recorded after training (episode.npz, written by tools/record_run.py): the
MuJoCo state before every one of its 1000 control steps (qpos, qvel, mocap_pos,
mocap_quat) and the rewards.  Each task is rendered from the first camera of its
model, as dm_control does, except where CAMERAS says otherwise, in real time: every
frame is one control step, or every second one for the tasks that step at 100 Hz.

    python tools/render_dmc_episodes.py <results> [--size 352] [--only Task ...]

<results>/<Task>/s<seed>/episode.npz; of several runs of a task, the one whose 64
episodes have the highest mean return is used, so that the video shows a policy that
solves the task, not a lucky episode of one that does not; that mean is reported with it.  Needs mujoco, mujoco_playground, imageio and imageio-ffmpeg.
"""
import argparse
import glob
import json
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from mujoco_playground import registry

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "assets", "rl")
# BallInCup: a free camera centred on the range of the cup and ball (the first camera of
# the model looks over the top of the cup); FishSwim: the side view that tracks the fish
CAMERAS = {
    "BallInCup": {"lookat": (0.0, 0.0, 0.53), "distance": 1.0, "azimuth": 90.0, "elevation": -4.0},
    "FishSwim": "tracking_y",
}


def best_episode(results, task):
    """-> (return of the episode, mean return of the 64, seed, episode) of the best run."""
    eps = []
    for path in glob.glob(os.path.join(results, task, "s*", "episode.npz")):
        z = np.load(path)
        eps.append((float(z["rewards"].sum()), float(z["returns"].mean()), int(os.path.basename(os.path.dirname(path))[1:]), z))
    return max(eps, key=lambda e: e[1]) if eps else None


def camera_for(model, task):
    spec = CAMERAS.get(task, 0)
    if isinstance(spec, dict):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = spec["lookat"]
        cam.distance, cam.azimuth, cam.elevation = spec["distance"], spec["azimuth"], spec["elevation"]
        return cam
    return model.camera(spec).id if isinstance(spec, str) else spec


def render(task, ep, size):
    model = registry.load(task).mj_model
    camera = camera_for(model, task)
    data = mujoco.MjData(model)
    dt = registry.load(task).dt
    stride = max(1, round(0.02 / dt))  # about 50 frames per second or fewer, in real time
    frames = []
    with mujoco.Renderer(model, height=size, width=size) as r:
        for t in range(0, ep["qpos"].shape[0], stride):
            data.qpos[:], data.qvel[:] = ep["qpos"][t], ep["qvel"][t]
            if model.nmocap:
                data.mocap_pos[:], data.mocap_quat[:] = ep["mocap_pos"][t], ep["mocap_quat"][t]
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=camera)
            frames.append(r.render().copy())
    return frames, 1.0 / (dt * stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--size", type=int, default=352)  # a multiple of 16, for H.264
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    index_path = os.path.join(OUT, "episodes.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    tasks = sorted(os.listdir(args.results)) if not args.only else args.only
    for task in tasks:
        found = best_episode(args.results, task)
        if found is None:
            print(f"{task}: no episode")
            continue
        ret, mean, seed, ep = found
        frames, fps = render(task, ep, args.size)
        mp4 = os.path.join(OUT, f"{task}.mp4")
        imageio.mimwrite(mp4, frames, fps=fps, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=16,
                         output_params=["-crf", "28", "-preset", "slow", "-movflags", "+faststart"])
        imageio.imwrite(os.path.join(OUT, f"{task}.jpg"), frames[len(frames) // 3], quality=82)
        index[task] = {"return": round(ret, 1), "mean": round(mean, 1), "episodes": int(ep["returns"].size), "seed": seed, "fps": round(fps, 2), "frames": len(frames),
                       "seconds": round(len(frames) / fps, 2)}
        print(f"{task}: return {ret:.1f}, mean of {ep['returns'].size} {mean:.1f} (seed {seed}), {len(frames)} frames at {fps:.0f} fps, "
              f"{os.path.getsize(mp4) / 1e3:.0f} kB")
    with open(index_path, "w") as f:
        json.dump(dict(sorted(index.items())), f, indent=1)


if __name__ == "__main__":
    main()
