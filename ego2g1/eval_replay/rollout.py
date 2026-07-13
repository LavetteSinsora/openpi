"""Phase 1 (headless box): teacher-forced policy rollout -> eval_rollout.npz.

Runs the checkpoint through the EXACT deployment stack (ego2g1.policy.create_policy
-> Policy.infer -> sample_actions Euler integration) on the real recorded observations,
and dumps the raw action chunks + anchor states. No mujoco/mink/display.

    uv run python -m ego2g1.eval_replay.rollout \
        --checkpoint checkpoints/ego2g1_pi05/run1/10000 \
        --source-episode put_bottle_in_box/episode_10 \
        --dataset-root /path/to/put_bottle_in_box --out eval_rollout.npz

`--synthetic-gt` skips the policy and writes a dump whose composed targets exactly
reproduce the recorded trajectory — a transform-correctness baseline (the eval robot
should overlay the ground-truth robot) that needs no checkpoint or jax.
"""

import argparse
import pathlib

import numpy as np

from ego2g1.eval_replay import dataset_io as dio

# action layout mirrors the state layout: per hand [eef vec9 (9) | hand cmd (6)]
A_EEF = {"left": slice(0, 9), "right": slice(15, 24)}
A_HAND = {"left": slice(9, 15), "right": slice(24, 30)}


def _resolve_episode(root, args) -> int:
    if args.episode is not None:
        return args.episode
    idx = dio.source_to_episodes(root, args.source_episode)
    if len(idx) > 1:
        print(f"NOTE: source {args.source_episode} -> LeRobot episodes {idx}; using {idx[0]} "
              "(one sub-episode = one timeline). Pass --episode to pick another.")
    return idx[0]


def _synthetic_gt_actions(ep: dio.EpisodeData, query_ticks, horizon) -> np.ndarray:
    """Actions that reconstruct the recorded label exactly: eef delta =
    inv(pose_anchor) @ pose_{t+1+k}, hand = hand[t+1+k] (held past the end)."""
    from ego2g1.chunk_math import se3_to_vec9, vec9_to_se3

    T = ep.n_frames
    hand = {"left": ep.hand_left, "right": ep.hand_right}
    out = np.zeros((len(query_ticks), horizon, 30), np.float32)
    for q, t in enumerate(query_ticks):
        for side in ("left", "right"):
            anchor = vec9_to_se3(ep.state[t][dio.EEF[side]])
            inv_anchor = np.linalg.inv(anchor)
            for k in range(horizon):
                j = min(t + 1 + k, T - 1)
                delta = se3_to_vec9(inv_anchor @ vec9_to_se3(ep.state[j][dio.EEF[side]]))
                out[q, k, A_EEF[side]] = delta
                out[q, k, A_HAND[side]] = hand[side][j]
    return out


def _policy_actions(ep, query_ticks, horizon, action_dim, args) -> np.ndarray:
    import jax

    from ego2g1 import policy as _policy

    p = _policy.create_policy(args.checkpoint, default_prompt=ep.task, assets_dir=args.assets_dir)
    p._sample_kwargs = {"num_steps": args.num_steps}  # deployment denoise steps  # noqa: SLF001
    base_key = jax.random.key(args.seed)
    frames = dio.read_video_frames_at(ep.video_path, query_ticks, ep.fps)  # one decode pass
    out = np.zeros((len(query_ticks), horizon, 30), np.float32)
    for q, t in enumerate(query_ticks):
        obs = {"observation/image": frames[q], "observation/state": ep.state[t], "prompt": ep.task}
        noise = np.asarray(jax.random.normal(jax.random.fold_in(base_key, q), (horizon, action_dim)))
        result = p.infer(obs, noise=noise)
        actions = np.asarray(result["actions"])  # (horizon, 30) raw units
        out[q] = actions
        print(f"  query {q + 1}/{len(query_ticks)} tick {t}: infer {result['policy_timing']['infer_ms']:.0f} ms")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", help="step dir, e.g. checkpoints/ego2g1_pi05/run1/10000")
    ap.add_argument("--assets-dir", default=None,
                    help="dir with norm_stats.json + per_slot_stats.npz. Default: the checkpoint's own "
                         "copies, else the training assets dir assets/<name>/<repo_id> (run from the "
                         "openpi root). Pass explicitly to override.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--episode", type=int, help="LeRobot episode index")
    ap.add_argument("--source-episode", help="e.g. put_bottle_in_box/episode_10 (first sub-ep used)")
    ap.add_argument("--stride", type=int, default=25, help="K executed actions per query before re-observing")
    ap.add_argument("--num-steps", type=int, default=10, help="flow-matching Euler denoise steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_rollout.npz")
    ap.add_argument("--synthetic-gt", action="store_true",
                    help="skip the policy; write a GT-reproducing dump (transform sanity, no checkpoint)")
    args = ap.parse_args()

    root = pathlib.Path(args.dataset_root)
    if args.episode is None and args.source_episode is None:
        ap.error("give --episode or --source-episode")
    episode_index = _resolve_episode(root, args)
    ep = dio.load_episode(root, episode_index)
    query_ticks = list(range(0, ep.n_frames, args.stride))
    print(f"episode {episode_index} ({ep.source_episode}), {ep.n_frames} frames, "
          f"K={args.stride} -> {len(query_ticks)} queries")

    horizon, action_dim = 50, 32
    provenance = {"extraction_config_hash": None, "ego2g1_config_hash": None}
    if args.synthetic_gt:
        actions = _synthetic_gt_actions(ep, query_ticks, horizon)
        mode = "synthetic_gt"
    else:
        if not args.checkpoint:
            ap.error("--checkpoint required unless --synthetic-gt")
        from ego2g1 import policy as _policy
        from ego2g1 import stamp as _stamp
        stamp = _stamp.check_supported(_policy.resolve_run_dir(args.checkpoint))
        cfg = _policy.config_from_stamp(stamp)
        horizon, action_dim = cfg.action_horizon, cfg.action_dim
        provenance = {"extraction_config_hash": stamp.get("extraction_config_hash"),
                      "ego2g1_config_hash": stamp.get("ego2g1_config_hash")}
        actions = _policy_actions(ep, query_ticks, horizon, action_dim, args)
        mode = "policy"

    anchor_state = ep.state[np.asarray(query_ticks)]  # (n_q, 30)
    out_path = pathlib.Path(args.out)
    np.savez(
        out_path,
        mode=mode,
        episode_index=episode_index,
        source_episode=ep.source_episode,
        task=ep.task,
        n_frames=ep.n_frames,
        query_ticks=np.asarray(query_ticks, np.int64),
        stride=args.stride,
        horizon=horizon,
        num_steps=args.num_steps,
        seed=args.seed,
        actions=actions.astype(np.float32),
        anchor_state=anchor_state.astype(np.float32),
        **provenance,
    )
    print(f"wrote {out_path}  (mode={mode}, actions {actions.shape})")


if __name__ == "__main__":
    main()
