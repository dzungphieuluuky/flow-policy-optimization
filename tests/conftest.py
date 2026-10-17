import json
import sys
from pathlib import Path

SNAPSHOT = Path(__file__).with_name("guard_messages.json")

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


def pytest_addoption(parser):
    parser.addoption(
        "--snapshot-regen",
        action="store_true",
        default=False,
        help="Rewrite tests/guard_messages.json from the live messages.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "snapshot_regen: regenerates a message snapshot")


def _discard_mujoco_log() -> None:
    """Remove a MUJOCO_LOG.TXT that this session created."""
    if _mujoco_log_existed_at_start or not MUJOCO_LOG.exists():
        return
    try:
        MUJOCO_LOG.unlink()
    except OSError as exc:
        print(f"\nleft {MUJOCO_LOG} in place: {exc}")


def pytest_sessionfinish(session, exitstatus):
    """Clear the MuJoCo log, then honour ``--snapshot-regen``.

    The cleanup runs first and does not return early, so the snapshot rewrite
    still happens when the flag is absent.
    """
    _discard_mujoco_log()
    if not session.config.getoption("--snapshot-regen"):
        return
    mod = sys.modules.get("test_guard_messages")
    if mod is None:
        return
    try:
        SNAPSHOT.write_text(
            json.dumps(mod._collect(), indent=2, sort_keys=True) + "\n"
        )
        print(f"\nwrote {SNAPSHOT} ({len(mod._CASES)} messages)")
    except Exception as exc:
        raise AssertionError(f"snapshot regen failed: {exc}") from exc
