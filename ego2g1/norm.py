"""Norm-stats artifacts: pooled openpi NormStats + the E001 per-slot grid.

Two files per stats computation, both stamped with provenance
(TRAINING_PLAN.md §3.6):
- `norm_stats.json` — openpi-native pooled per-dim stats for `state` and
  `actions` (written via openpi.shared.normalize.save); consumed unchanged by
  stock Normalize/Unnormalize and the checkpoint assets hook.
- `per_slot_stats.npz` — sigma_slot (H, D_real) + provenance json; the E001
  gain grid is DERIVED at load time for the configured floor c (sigma is the
  artifact, c is config).
"""

import dataclasses
import json
import pathlib

import numpy as np

import openpi.shared.normalize as _normalize

PER_SLOT_FILENAME = "per_slot_stats.npz"

# (slot, dim) pairs allowed to be degenerate (q99-q01 or sigma_slot ~ 0).
# rot6d identity-adjacent dims at slot 1 and unused left fingers are known;
# extend deliberately, never silently. Checked per-dim over ALL slots for
# pooled stats and per (slot, dim) for sigma_slot.
DEGENERATE_EPS = 1e-8


@dataclasses.dataclass(frozen=True)
class PerSlotStats:
    sigma_slot: np.ndarray  # (H, D_real)
    provenance: dict

    def gain(self, floor_c: float, sigma_pooled: np.ndarray) -> np.ndarray:
        """E001: gain[k,d] = sigma_pooled[d] / max(sigma_slot[k,d], c*sigma_pooled[d]).
        floor_c=1 -> gain==1 (bitwise stock pooled behavior). Dims whose pooled
        sigma is itself degenerate get gain 1 (they carry no signal at all)."""
        if not 0 < floor_c <= 1:
            raise ValueError(f"floor_c={floor_c} must be in (0, 1]")
        sp = np.asarray(sigma_pooled, dtype=np.float64)[: self.sigma_slot.shape[1]]
        divisor = np.maximum(self.sigma_slot, floor_c * sp[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            g = sp[None, :] / divisor
        g = np.where(sp[None, :] <= DEGENERATE_EPS, 1.0, g)
        # gain <= 1/c by the floor; gain < 1 is legitimate (late slots whose
        # sigma exceeds the pooled sigma get shrunk toward unit scale).
        assert np.all(np.isfinite(g)) and np.all(g > 0.0) and np.all(g <= 1.0 / floor_c + 1e-9)
        return g.astype(np.float32)


def save_per_slot(directory: pathlib.Path | str, stats: PerSlotStats) -> None:
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / PER_SLOT_FILENAME,
        sigma_slot=stats.sigma_slot,
        provenance=json.dumps(stats.provenance),
    )


def load_per_slot(directory: pathlib.Path | str) -> PerSlotStats:
    path = pathlib.Path(directory) / PER_SLOT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m ego2g1.compute_norm_stats` (E001 needs the per-slot grid)"
        )
    with np.load(path, allow_pickle=False) as z:
        return PerSlotStats(
            sigma_slot=np.asarray(z["sigma_slot"], dtype=np.float64),
            provenance=json.loads(str(z["provenance"])),
        )


def load_pooled(directory: pathlib.Path | str) -> dict[str, _normalize.NormStats]:
    return _normalize.load(directory)


def check_stats_sanity(
    pooled: dict[str, _normalize.NormStats],
    per_slot: PerSlotStats,
    degenerate_dim_allowlist: tuple[int, ...],
) -> list[str]:
    """E001 eval item 7. Returns human-readable violations (empty = pass).
    Degenerate pooled action dims and all-slot-degenerate sigma rows must be
    in the allowlist; anything else is a data bug, not a tuning knob."""
    problems = []
    act = pooled["actions"]
    d_real = per_slot.sigma_slot.shape[1]
    for d in range(d_real):
        span = float(act.q99[d] - act.q01[d])
        if span <= DEGENERATE_EPS and d not in degenerate_dim_allowlist:
            problems.append(f"actions dim {d}: pooled q99-q01 = {span:.3e} (degenerate, not allowlisted)")
    for d in range(d_real):
        if d in degenerate_dim_allowlist:
            continue
        # slot 0 of anchor-relative deltas is legitimately tiny; require SOME
        # signal by mid-chunk rather than at every slot.
        if float(per_slot.sigma_slot[:, d].max()) <= DEGENERATE_EPS:
            problems.append(f"actions dim {d}: sigma_slot ~ 0 at every slot (not allowlisted)")
    return problems
