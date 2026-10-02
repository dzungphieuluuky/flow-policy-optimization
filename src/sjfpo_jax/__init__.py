"""SJFPO in JAX. Milestone 1: the FPO++ core math, the DCT, and chunk-level GAE.

Mirrors ``spectral_joint_policy``; nothing here imports torch or lerobot. The
torch package stays the oracle and the parity tests read both sides.

x64 is on because ``RunningMeanStd.count`` must stay fp64: it dies at 2^24
frames in fp32 (``networks.py:4``). JAX truncates a float64 request to float32
with only a warning, so this has to be set before the first array exists.
"""
import jax

jax.config.update("jax_enable_x64", True)
