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


def eef_proprioception_mse(ep, eval_arm, renderer):
    """Per-frame MSE between the eval robot's EEF proprioception and GT's, over
    both hands' flange vec9 (18 dims). Eval EEF is FK'd from eval_arm and
    expressed in the pelvis frame (same convention as the recorded `state` EEF).
    Expected shape: ~0 at each teacher-forcing re-sync, growing between."""
    import mujoco

    from ego2g1.chunk_math import se3_to_vec9

    be = renderer.backend
    mse = np.zeros(ep.n_frames)
    for t in range(ep.n_frames):
        be.data.qpos[renderer.arm_adr] = eval_arm[t]
        mujoco.mj_forward(be.model, be.data)
        err = 0.0
        for side in ("left", "right"):
            eval_vec9 = se3_to_vec9(be.world_to_base(be.flange_pose(side)))
            err += float(np.mean((eval_vec9 - ep.state[t][dio.EEF[side]]) ** 2))
        mse[t] = err / 2.0
    return mse


def _timeline_base(mse, query_ticks, width, height=150):
    """Static timeline: MSE curve + green teacher-forcing markers + axes."""
    import cv2

    img = np.full((height, width, 3), 22, np.uint8)
    left, right, top, bot = 66, 12, 26, 22
    pw, ph = width - left - right, height - top - bot
    T = len(mse)
    vmax = max(float(mse.max()), 1e-9)
    x = lambda t: int(left + (t / max(T - 1, 1)) * pw)
    y = lambda v: int(top + ph * (1.0 - v / vmax))

    cv2.putText(img, "EEF proprioception MSE  (eval vs ground truth)", (left, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "teacher-forcing re-sync", (width - 230, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 210, 120), 1, cv2.LINE_AA)
    # axes
    cv2.line(img, (left, top), (left, top + ph), (90, 90, 90), 1)
    cv2.line(img, (left, top + ph), (left + pw, top + ph), (90, 90, 90), 1)
    cv2.putText(img, f"{vmax:.2g}", (4, top + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(img, "0", (4, top + ph), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    # teacher-forcing markers (green verticals where eval is restored to GT)
    for qt in query_ticks:
        if 0 <= qt < T:
            cv2.line(img, (x(qt), top), (x(qt), top + ph), (60, 120, 70), 1)
    # MSE curve
    pts = np.array([[x(t), y(mse[t])] for t in range(T)], np.int32)
    cv2.polylines(img, [pts], False, (90, 170, 240), 2, cv2.LINE_AA)
    inv = lambda px: int(np.clip(round((px - left) / max(pw, 1) * (T - 1)), 0, T - 1))
    return img, x, inv


def _timeline_frame(base, xfn, t, height):
    import cv2
    img = base.copy()
    cx = xfn(t)
    cv2.line(img, (cx, 24), (cx, height - 22), (255, 255, 255), 1)
    return img


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


def run_interactive(ep, eval_arm, eval_hl, eval_hr, query_ticks, mount, fps=30,
                    data_extraction_root=None):
    """Orbitable mujoco viewer: GT (left) + eval (right) in one scene. Drag to
    rotate/zoom/pan; SPACE play/pause, left/right arrow step. Loops."""
    import time

    import mujoco
    import mujoco.viewer

    from ego2g1.eval_replay.scene import TwoRobotScene

    scene = TwoRobotScene(hand_mount=mount, data_extraction_root=data_extraction_root)
    T = ep.n_frames
    state = {"t": 0, "paused": False}

    def key_cb(keycode):
        if keycode == 32:  # SPACE
            state["paused"] = not state["paused"]
        elif keycode == 262:  # right arrow
            state["t"] = min(state["t"] + 1, T - 1); state["paused"] = True
        elif keycode == 263:  # left arrow
            state["t"] = max(state["t"] - 1, 0); state["paused"] = True

    print("interactive: drag to orbit · SPACE play/pause · <-/-> step · GT is left, eval is right")
    try:
        cm = mujoco.viewer.launch_passive(scene.model, scene.data, key_callback=key_cb)
    except RuntimeError as e:
        if "mjpython" in str(e):
            raise SystemExit(
                "macOS needs mjpython for the interactive viewer. Re-run with:\n"
                "  PYTHONPATH=. ../../.venv/bin/mjpython -m ego2g1.eval_replay.viewer <same args>\n"
                "(or drop --interactive to use the OpenCV window / --mp4)."
            ) from e
        raise
    with cm as v:
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.4, 150, -12
        v.cam.lookat[:] = [0.0, -0.45, 1.0]
        while v.is_running():
            t = state["t"]
            scene.set_frame(ep.arm_qpos[t], ep.hand_left[t], ep.hand_right[t],
                            eval_arm[t], eval_hl[t], eval_hr[t])
            v.sync()
            time.sleep(1.0 / fps)
            if not state["paused"]:
                state["t"] = 0 if t >= T - 1 else t + 1


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
    ap.add_argument("--interactive", action="store_true",
                    help="orbitable mujoco 3D viewer with both robots (drag to rotate); needs a display")
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

    mount = {"xyz": args.hand_mount_xyz or [0, 0, 0], "rpy": args.hand_mount_rpy or [0, 0, 0]} \
        if (args.hand_mount_xyz or args.hand_mount_rpy) else None
    query_ticks = dump["query_ticks"].tolist()

    # IK reconstruction of the eval trajectory (arms-only backend is enough).
    ik_r = G1Renderer(with_hands=False)
    print("reconstructing eval trajectory (IK)...")
    eval_arm, eval_hl, eval_hr = reconstruct_eval(dump, ep, ik_r)
    mse = eef_proprioception_mse(ep, eval_arm, ik_r)

    if args.interactive:
        run_interactive(ep, eval_arm, eval_hl, eval_hr, query_ticks, mount,
                        data_extraction_root=args.data_extraction_path)
        return

    video = dio.read_video_frames(ep.video_path, ep.n_frames, ep.fps)
    T = min(ep.n_frames, len(video))
    gt_r = G1Renderer(with_hands=args.hands, hand_mount=mount,
                      data_extraction_root=args.data_extraction_path)
    eval_r = G1Renderer(with_hands=args.hands, hand_mount=mount,
                        data_extraction_root=args.data_extraction_path)
    print("rendering frames...")
    comps = []
    for t in range(T):
        gt_r.set_pose(ep.arm_qpos[t], ep.hand_left[t], ep.hand_right[t])
        eval_r.set_pose(eval_arm[t], eval_hl[t], eval_hr[t])
        step = "(anchor=GT)" if t in query_ticks else "(predicted)"
        comps.append(compose_frame(gt_r.render(), eval_r.render(), video[t], t, step,
                                   (ep.hand_left[t], ep.hand_right[t]), (eval_hl[t], eval_hr[t]), h=args.height))
    comp_w = comps[0].shape[1]
    tl_base, xfn, inv_xfn = _timeline_base(mse, query_ticks, comp_w)
    tl_h = tl_base.shape[0]
    frames = np.stack([
        cv2.cvtColor(np.concatenate([comp, _timeline_frame(tl_base, xfn, t, tl_h)], axis=0), cv2.COLOR_RGB2BGR)
        for t, comp in enumerate(comps)
    ])
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
    fps = 30
    comp_h = H - tl_h  # timeline strip occupies the bottom tl_h rows

    def on_mouse(event, x, y, flags, _param):
        dragging = event == cv2.EVENT_LBUTTONDOWN or (
            event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON)
        if dragging and y >= comp_h:  # drag anywhere on the timeline strip -> scrub live
            state["t"] = inv_xfn(x)
            state["playing"] = False
            cv2.setTrackbarPos("frame", win, state["t"])

    cv2.setMouseCallback(win, on_mouse)
    state["playing"] = False
    print("controls: SPACE play/pause · a/d (or ,/.) step · DRAG ON THE TIMELINE to scrub · q/ESC quit")
    while True:
        # poll the trackbar too (its callback only fires on mouse-up on macOS)
        pos = cv2.getTrackbarPos("frame", win)
        if not state["playing"] and pos != state["t"]:
            state["t"] = pos
        playing = state["playing"]
        t = state["t"]
        cv2.imshow(win, frames[t])
        key = cv2.waitKey(max(1, int(1000 / fps)) if playing else 15) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == 32:  # SPACE
            state["playing"] = not state["playing"]
        elif key in (ord("d"), ord("."), 83):  # step forward
            state["t"] = min(t + 1, T - 1); state["playing"] = False
            cv2.setTrackbarPos("frame", win, state["t"])
        elif key in (ord("a"), ord(","), 81):  # step back
            state["t"] = max(t - 1, 0); state["playing"] = False
            cv2.setTrackbarPos("frame", win, state["t"])
        if state["playing"]:
            nxt = 0 if t >= T - 1 else t + 1  # loop at the end
            state["t"] = nxt
            cv2.setTrackbarPos("frame", win, nxt)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
