"""Checkpoint feature stamping + load guard (OPENPI_EDITS.md E002 item 5).

Ego2G1 checkpoints have a stock-shaped param tree, so stock openpi would load
and serve them with silently wrong semantics (missing per-slot inverse
rescale, missing control-mode prompt, ...). The stamp makes that loud: every
checkpoint records its feature flags + provenance; loading refuses unless the
running code declares support for every flag marked `required`.
"""

import dataclasses
import json
import pathlib
import subprocess

STAMP_FILENAME = "ego2g1_stamp.json"

# Features this codebase knows how to serve. A checkpoint requiring anything
# outside this set (e.g. from a newer ego2g1) must be refused.
SUPPORTED_FEATURES = frozenset({
    "per_slot_rescale",
    "per_slot_center",
    "degenerate_neutralization",
    "model_space_clamp",
    "control_mode_prompt",
    "relative_chunk_actions",
    "action_dim_actual",
    "rtc_training",
    "state_masking",
})


class UnsupportedCheckpointError(RuntimeError):
    pass


def _git_commit(repo_dir) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_stamp(checkpoint_dir, train_config, extraction_config_hash: str | None) -> None:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    openpi_root = pathlib.Path(__file__).resolve().parent.parent
    stamp = {
        "feature_flags": train_config.feature_flags(),
        "ego2g1_config": dataclasses.asdict(train_config),
        "ego2g1_config_hash": train_config.config_hash(),
        "extraction_config_hash": extraction_config_hash,
        "openpi_commit": _git_commit(openpi_root),
    }
    (checkpoint_dir / STAMP_FILENAME).write_text(json.dumps(stamp, indent=2, default=str))


def read_stamp(checkpoint_dir) -> dict:
    path = pathlib.Path(checkpoint_dir) / STAMP_FILENAME
    if not path.exists():
        raise UnsupportedCheckpointError(
            f"{path} missing: not an ego2g1 checkpoint (or written before stamping existed). "
            "Refusing to guess its serving semantics."
        )
    return json.loads(path.read_text())


def check_supported(checkpoint_dir) -> dict:
    """Load the stamp and refuse to serve unknown required features."""
    stamp = read_stamp(checkpoint_dir)
    flags = stamp["feature_flags"]
    unknown_required = [
        name for name, spec in flags.items()
        if isinstance(spec, dict) and spec.get("required") and name not in SUPPORTED_FEATURES
    ]
    if unknown_required:
        raise UnsupportedCheckpointError(
            f"Checkpoint {checkpoint_dir} requires features this code does not support: {unknown_required}"
        )
    return stamp
