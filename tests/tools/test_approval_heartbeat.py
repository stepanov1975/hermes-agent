"""Tests for the activity-heartbeat behavior of the blocking gateway approval wait.

Regression test for false gateway inactivity timeouts firing while the agent
is legitimately blocked waiting for a user to respond to a dangerous-command
approval prompt.  Before the fix, ``entry.event.wait(timeout=...)`` blocked
silently — no ``_touch_activity()`` calls — and the gateway's inactivity
watchdog (``agent.gateway_timeout``, default 1800s) would kill the agent
while the user was still choosing whether to approve.

The fix polls the event in short slices and fires ``touch_activity_if_due``
between slices, mirroring ``_wait_for_process`` in ``tools/environments/base.py``.
"""

import os
import threading
import time


def _clear_approval_state():
    """Reset all module-level approval state between tests."""
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


class TestApprovalHeartbeat:
    """The blocking gateway approval wait must fire activity heartbeats.

    Without heartbeats, the gateway's inactivity watchdog kills the agent
    thread while it's legitimately waiting for a slow user to respond to
    an approval prompt (observed in real user logs: MRB, April 2026).
    """

    SESSION_KEY = "heartbeat-test-session"

    def setup_method(self):
        _clear_approval_state()
        self._saved_env = {
            k: os.environ.get(k)
            for k in ("HERMES_GATEWAY_SESSION", "HERMES_YOLO_MODE",
                      "HERMES_SESSION_KEY", "HERMES_CRON_SESSION")
        }
        os.environ.pop("HERMES_YOLO_MODE", None)
        os.environ.pop("HERMES_CRON_SESSION", None)
        os.environ["HERMES_GATEWAY_SESSION"] = "1"
        # The blocking wait path reads the session key via contextvar OR
        # os.environ fallback.  Contextvars don't propagate across threads
        # by default, so env var is the portable way to drive this in tests.
        os.environ["HERMES_SESSION_KEY"] = self.SESSION_KEY

    def teardown_method(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _clear_approval_state()

    def test_gateway_wait_uses_gateway_timeout_not_cli_timeout(self, monkeypatch):
        """Gateway approvals must not inherit the short CLI prompt timeout."""
        from tools import approval as mod

        monkeypatch.setattr(
            mod,
            "_get_approval_config",
            lambda: {"mode": "manual", "timeout": 0, "gateway_timeout": 2},
        )
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: None)

        result_holder = {}

        def _check():
            result_holder["result"] = mod.check_all_command_guards(
                "rm -rf .git", "local"
            )

        thread = threading.Thread(target=_check)
        thread.start()

        for _ in range(50):
            if mod._gateway_queues.get(self.SESSION_KEY):
                break
            time.sleep(0.02)

        assert mod._gateway_queues.get(self.SESSION_KEY), "approval request was not queued"
        mod.resolve_gateway_approval(self.SESSION_KEY, "once")
        thread.join(timeout=5)

        assert "result" in result_holder, "approval wait did not return after approve"
        assert result_holder["result"]["approved"] is True
        assert result_holder["result"].get("user_approved") is True

    def test_gateway_wait_polls_activity_while_pending(self, monkeypatch):
        """A pending gateway approval must refresh activity while it waits."""
        from tools import approval as mod
        from tools.environments import base as env_base

        monkeypatch.setattr(
            mod,
            "_get_approval_config",
            lambda: {"mode": "manual", "timeout": 60, "gateway_timeout": 2},
        )
        touches = []

        def _touch(state, label):
            touches.append((label, round(time.monotonic() - state["start"], 1)))

        monkeypatch.setattr(env_base, "touch_activity_if_due", _touch)
        mod.register_gateway_notify(self.SESSION_KEY, lambda data: None)

        result = mod.check_all_command_guards("rm -rf .git", "local")

        assert result["approved"] is False
        assert result.get("outcome") == "timeout"
        assert touches, "approval wait did not call touch_activity_if_due while pending"
        assert all(label == "waiting for user approval" for label, _ in touches)
