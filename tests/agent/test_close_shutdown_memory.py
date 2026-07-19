"""Tests for AIAgent.close() calling shutdown_memory_provider (#46082)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from plugins.memory.honcho import HonchoMemoryProvider
from run_agent import AIAgent


class _FakeMemoryManager:
    """Track memory-provider lifecycle calls."""

    def __init__(self) -> None:
        self.received_messages: list[list[dict]] = []
        self.shutdown_calls = 0

    def on_session_end(self, messages: list[dict]) -> None:
        self.received_messages.append(messages)

    def shutdown_all(self) -> None:
        self.shutdown_calls += 1

    def on_session_switch(self, *args, **kwargs) -> None:
        pass


class _FakeCompressor:
    def __init__(self) -> None:
        self.received_messages: list[list[dict]] = []

    def on_session_end(self, sid: str, messages: list[dict]) -> None:
        self.received_messages.append(messages)


def _agent() -> tuple[AIAgent, _FakeMemoryManager, _FakeCompressor]:
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "test-close-shutdown"
    agent._session_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._memory_provider_shutdown_condition = threading.Condition()
    agent._memory_provider_shutdown_state = "not_started"
    agent._memory_provider_shutdown_owner = None
    agent.client = None
    agent._end_session_on_close = False
    agent.background_review_callback = None
    agent.memory_notifications = "on"
    manager = _FakeMemoryManager()
    compressor = _FakeCompressor()
    agent._memory_manager = manager
    agent.context_compressor = compressor
    return agent, manager, compressor


def test_close_calls_shutdown_memory_provider() -> None:
    """close() must run the provider and context-engine shutdown chain."""
    agent, manager, compressor = _agent()

    agent.close()

    assert len(manager.received_messages) == 1
    assert manager.shutdown_calls == 1
    assert len(compressor.received_messages) == 1


def test_close_is_idempotent_with_memory_provider() -> None:
    """Explicit shutdown followed by repeated close tears resources down once."""
    agent, manager, compressor = _agent()

    agent.shutdown_memory_provider()
    agent.close()
    agent.close()

    assert manager.shutdown_calls == 1
    assert len(manager.received_messages) == 1
    assert len(compressor.received_messages) == 1


def test_close_passes_non_empty_messages_to_providers() -> None:
    """close() must notify providers before clearing the transcript."""
    agent, manager, compressor = _agent()
    expected = list(agent._session_messages)

    agent.close()

    assert manager.received_messages == [expected]
    assert compressor.received_messages == [expected]
    assert agent._session_messages == []


def test_no_arg_shutdown_uses_current_transcript() -> None:
    """Legacy explicit callers must not accidentally finalize an empty session."""
    agent, manager, compressor = _agent()
    expected = list(agent._session_messages)

    agent.shutdown_memory_provider()

    assert manager.received_messages == [expected]
    assert compressor.received_messages == [expected]


def test_concurrent_shutdown_waits_for_provider_cleanup() -> None:
    agent, manager, _compressor = _agent()
    entered = threading.Event()
    release = threading.Event()

    def blocking_shutdown() -> None:
        entered.set()
        assert release.wait(timeout=2)
        manager.shutdown_calls += 1

    manager.shutdown_all = blocking_shutdown  # type: ignore[method-assign]
    first = threading.Thread(target=agent.shutdown_memory_provider)
    second_done = threading.Event()

    def second_shutdown() -> None:
        agent.shutdown_memory_provider()
        second_done.set()

    second = threading.Thread(target=second_shutdown)
    first.start()
    assert entered.wait(timeout=2)
    second.start()
    assert not second_done.wait(timeout=0.05)
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_done.is_set()
    assert manager.shutdown_calls == 1


def test_same_thread_reentry_does_not_deadlock_or_repeat_shutdown() -> None:
    agent, manager, compressor = _agent()

    def reentrant_on_session_end(messages: list[dict]) -> None:
        manager.received_messages.append(messages)
        agent.shutdown_memory_provider(messages)

    manager.on_session_end = reentrant_on_session_end  # type: ignore[method-assign]

    agent.shutdown_memory_provider()

    assert manager.shutdown_calls == 1
    assert len(manager.received_messages) == 1
    assert len(compressor.received_messages) == 1


def test_honcho_shutdown_stops_session_manager_writer() -> None:
    provider = HonchoMemoryProvider.__new__(HonchoMemoryProvider)
    provider._prefetch_thread = None
    provider._sync_thread = None
    provider._init_thread = None
    provider._session_initialized = True
    provider._manager = MagicMock()

    provider.shutdown()

    provider._manager.shutdown.assert_called_once_with()
    provider._manager.flush_all.assert_called_once_with()
