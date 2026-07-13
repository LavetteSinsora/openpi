"""Pure-numpy math shared by training, serving, and the robot client.

Nothing here may import jax, openpi, mujoco, or DDS: `ego2g1.deploy` runs on a
robot PC that has none of them. The vec9/rot6d encoding is NOT re-implemented
here — it is re-exported from `ego2g1.chunk_math`, which is byte-pinned against
the training loader by data_extraction/tests/test_loader_equivalence.py. A third
copy would be a third thing to drift.
"""
