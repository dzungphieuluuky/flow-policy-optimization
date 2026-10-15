import json
import sys
from pathlib import Path

SNAPSHOT = Path(__file__).with_name("guard_messages.json")


def pytest_addoption(parser):
    parser.addoption(
        "--snapshot-regen",
        action="store_true",
        default=False,
        help="Rewrite tests/guard_messages.json from the live messages.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "snapshot_regen: regenerates a message snapshot")


def pytest_sessionfinish(session, exitstatus):
    """``--snapshot-regen`` rewrites the snapshot once, at the end of the run."""
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
