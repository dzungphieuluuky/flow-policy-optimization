from pathlib import Path

import torch

# MuJoCo writes its warnings to MUJOCO_LOG.TXT in the process working directory.
# Any test that builds a model leaves one there.
#
# The file cannot be redirected. MuJoCo names it from a C-side default, the
# Python binding does not expose MjOption.logfile, and the compiled module has
# no environment variable for it. Verified on mujoco 3.13.0: setting MUJOCO_LOG
# changed nothing, and the file appeared either way.
#
# So the suite removes the file instead of redirecting it. It removes it only
# when the file did not exist when the session began, so a log left by a real
# training run survives the suite.
MUJOCO_LOG = Path("MUJOCO_LOG.TXT")
_mujoco_log_existed_at_start = MUJOCO_LOG.exists()


def _discard_mujoco_log() -> None:
    """Remove a MUJOCO_LOG.TXT that this session created."""
    if _mujoco_log_existed_at_start or not MUJOCO_LOG.exists():
        return
    try:
        MUJOCO_LOG.unlink()
    except OSError as exc:
        print(f"\nleft {MUJOCO_LOG} in place: {exc}")


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001 - pytest names the hook
    """Clear the MuJoCo log the session created."""
    _discard_mujoco_log()


def unzero_critic_head(critic_or_member):
    """Kick every final trunk Linear off the zero-init by a tiny, deterministic
    amount.

    The critic's final head is zero-initialized on purpose (unbiased advantage
    at step 0), which makes raw Q constant across the board. The architecture
    tests (frame order, exec_length_frac gradient, member independence) ask
    about sensitivity, not the deployment init; they build a critic and probe
    it. Give those probes a non-degenerate head instead of weakening them.
    Accepts a Critic (all members) or a single CriticMember.
    """
    members = getattr(critic_or_member, "networks", [critic_or_member])
    with torch.no_grad():
        for i, member in enumerate(members):
            final = member.mlp[-1]
            if final.weight.abs().sum() > 0:
                continue
            gen = torch.Generator().manual_seed(1234 + i)
            final.weight.add_(0.05 * torch.randn(final.weight.shape, generator=gen))
            final.bias.add_(0.01)
