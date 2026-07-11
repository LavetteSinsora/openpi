"""ego2g1: self-contained pi05 fine-tuning package on top of stock openpi.

Everything that deviates from stock openpi lives here (TRAINING_PLAN.md in
the outer repo; OPENPI_EDITS.md for the per-deviation log). src/openpi is
bit-stock: model-behavior changes live in ego2g1.model (Pi0 subclass) and
ego2g1.gemma_patch (per-token adaRMS rebind), data-side changes in the
transforms/dataset/norm modules. Run entrypoints from the openpi root, e.g.
`uv run python -m ego2g1.train`.
"""
