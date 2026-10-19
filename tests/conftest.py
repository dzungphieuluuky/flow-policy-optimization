from pathlib import Path

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
