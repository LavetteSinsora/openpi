"""Phase 2 (display machine): reconstruct the eval trajectory from eval_rollout.npz
and render two G1s (ground truth + evaluated checkpoint) beside the egocentric video,
scrubbable on one timeline.

    python -m ego2g1.eval_replay.viewer --rollout eval_rollout.npz \
        --dataset-root /path/to/put_bottle_in_box \
        --data-extraction-path /path/to/ego-pi-replication [--hands]

Auto-detects a display: live OpenCV window + trackbar if $DISPLAY is set, else writes
a composited mp4. All rendering is offscreen (mujoco.Renderer), so it works headless.
"""

import argparse
import os
import pathlib
import sys

import numpy as np

from ego2g1.eval_replay import dataset_io as dio

A_EEF = {"left": slice(0, 9), "right": slice(15, 24)}
A_HAND = {"left": slice(9, 15), "right": slice(24, 30)}


def reconstruct_eval(dump, ep, ik_renderer):
    """Per-frame eval arm_qpos(14) + hand cmds, teacher-forced: query ticks stay
    at the GT anchor (re-sync snap); inter-anchor frames are the predicted arc."""
    from ego2g1.chunk_math import vec9_to_se3
    from ego2g1.eval_replay.scene import EvalIK

    T = ep.n_frames
    query_ticks = dump["query_ticks"].tolist()
    stride = int(dump["stride"])
    actions = dump["actions"]            # (n_q, H, 30)
    anchor_state = dump["anchor_state"]  # (n_q, 30)
    tick_set = set(query_ticks)

    eval_arm = ep.arm_qpos.copy()
    eval_hand = {"left": ep.hand_left.copy(), "right": ep.hand_right.copy()}
    ik = EvalIK(ik_renderer)

    for q, t in enumerate(query_ticks):
        ik.reset_to_arm(ep.arm_qpos[t])  # ground at the true (measured) state
        anchor = {s: vec9_to_se3(anchor_state[q][dio.EEF[s]]) for s in ("left", "right")}
        for k in range(stride):
            frame = t + 1 + k
            if frame >= T or (frame in tick_set and frame > t):
                break  # stop at the episode end or the next re-sync (that frame stays GT)
            tgt = {s: anchor[s] @ vec9_to_se3(actions[q, k, A_EEF[s]]) for s in ("left", "right")}
            eval_arm[frame] = ik.solve(tgt["left"], tgt["right"])
            for s in ("left", "right"):
                eval_hand[s][frame] = actions[q, k, A_HAND[s]]
    return eval_arm, eval_hand["left"], eval_hand["right"]


# --- compositing --------------------------------------------------------------

def _resize_h(img, h):
    import cv2
    w = int(round(img.shape[1] * h / img.shape[0]))
    return cv2.resize(img, (w, h))


def _label(img, text, grips=None):
    import cv2
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if grips is not None:  # (hand_left6, hand_right6): 12 small closure bars bottom-left
        y0 = img.shape[0] - 8
        for hi, cmd in enumerate(grips):
            for i, v in enumerate(cmd):
                x = 6 + hi * 100 + i * 15
                cv2.rectangle(img, (x, y0), (x + 11, y0 - int(28 * float(np.clip(v, 0, 1)))),
                              (80, 200, 120), -1)
                cv2.rectangle(img, (x, y0), (x + 11, y0 - 28), (120, 120, 120), 1)
    return img


def compose_frame(img_gt, img_eval, img_video, t, step_label, gt_hands, eval_hands, h=480):
    import cv2
    gt = _label(_resize_h(img_gt, h), "GROUND TRUTH", gt_hands)
    ev = _label(_resize_h(img_eval, h), f"EVAL  {step_label}", eval_hands)
    vid = _label(_resize_h(img_video, h), f"EGOCENTRIC (real)   frame {t}")
    strip = np.zeros((h, 4, 3), np.uint8)
    return np.concatenate([gt, strip, ev, strip, vid], axis=1)


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout", required=True, help="eval_rollout.npz from Phase 1")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--data-extraction-path", required=True,
                    help="path containing the data_extraction package (sim/hand/assets)")
    ap.add_argument("--hands", action="store_true", help="attach revo2 hands (else arms + grip bars only)")
    ap.add_argument("--hand-mount-xyz", type=float, nargs=3, default=None)
    ap.add_argument("--hand-mount-rpy", type=float, nargs=3, default=None)
    ap.add_argument("--out", default="eval_replay.mp4", help="mp4 written when headless / --mp4")
    ap.add_argument("--mp4", action="store_true", help="force mp4 output (no live window)")
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(args.data_extraction_path).resolve()))
    import cv2

    from ego2g1.eval_replay.scene import G1Renderer

    dump = np.load(args.rollout, allow_pickle=False)
    episode_index = int(dump["episode_index"])
    ep = dio.load_episode(args.dataset_root, episode_index)
    if ep.n_frames != int(dump["n_frames"]):
        raise ValueError(f"dataset episode has {ep.n_frames} frames, dump expects {int(dump['n_frames'])} "
                         "— dataset copy mismatch")
    print(f"episode {episode_index} ({ep.source_episode}) — {ep.n_frames} frames, mode={str(dump['mode'])}")

    video = dio.read_video_frames(ep.video_path, ep.n_frames)
    T = min(ep.n_frames, len(video))

    mount = None
    if args.hands:
        mount = {"xyz": args.hand_mount_xyz or [0, 0, 0], "rpy": args.hand_mount_rpy or [0, 0, 0]}
    gt_r = G1Renderer(with_hands=args.hands, hand_mount=mount)
    eval_r = G1Renderer(with_hands=args.hands, hand_mount=mount)

    print("reconstructing eval trajectory (IK)...")
    eval_arm, eval_hl, eval_hr = reconstruct_eval(dump, ep, eval_r)

    query_ticks = dump["query_ticks"].tolist()
    print("rendering frames...")
    frames = []
    for t in range(T):
        gt_r.set_pose(ep.arm_qpos[t], ep.hand_left[t], ep.hand_right[t])
        eval_r.set_pose(eval_arm[t], eval_hl[t], eval_hr[t])
        step = "(anchor=GT)" if t in query_ticks else "(predicted)"
        comp = compose_frame(gt_r.render(), eval_r.render(), video[t], t, step,
                             (ep.hand_left[t], ep.hand_right[t]), (eval_hl[t], eval_hr[t]), h=args.height)
        frames.append(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    frames = np.stack(frames)
    H, W = frames.shape[1:3]

    headless = args.mp4 or (not os.environ.get("DISPLAY") and sys.platform.startswith("linux"))
    if headless:
        out = pathlib.Path(args.out)
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"headless — wrote {out}  ({T} frames, {W}x{H})")
        return

    win = "ego2g1 eval replay  [GT | EVAL | egocentric]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    state = {"t": 0}
    cv2.createTrackbar("frame", win, 0, T - 1, lambda v: state.update(t=v))
    print("scrub the 'frame' trackbar; q or ESC to quit")
    while True:
        cv2.imshow(win, frames[state["t"]])
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
