"""Tests for unmute.quest_manager."""

import asyncio

import pytest

from unmute.exceptions import (
    MissingServiceAtCapacity,
    MissingServiceTimeout,
    WebSocketClosedError,
)
from unmute.quest_manager import Quest, QuestManager


# ---------------------------------------------------------------------------
# Quest
# ---------------------------------------------------------------------------


class TestQuest:
    @pytest.mark.asyncio
    async def test_lifecycle(self):
        """Init → run → close are called in order."""
        log: list[str] = []

        async def init():
            log.append("init")
            return 42

        async def run(data):
            log.append(f"run({data})")

        async def close(data):
            log.append(f"close({data})")

        q = Quest("test", init, run, close)
        async with q:
            # Wait for the task to finish running
            await asyncio.sleep(0.05)

        assert log == ["init", "run(42)", "close(42)"]

    @pytest.mark.asyncio
    async def test_get_returns_init_result(self):
        async def init():
            return "hello"

        async def run(data):
            await asyncio.sleep(10)

        q = Quest("test", init, run)
        async with q:
            result = await q.get()
            assert result == "hello"

    @pytest.mark.asyncio
    async def test_get_nowait_before_init(self):
        started = asyncio.Event()

        async def init():
            await started.wait()
            return 1

        async def run(_):
            await asyncio.sleep(10)

        q = Quest("test", init, run)
        async with q:
            assert q.get_nowait() is None
            started.set()

    @pytest.mark.asyncio
    async def test_get_nowait_after_init(self):
        async def init():
            return 99

        async def run(data):
            await asyncio.sleep(10)

        q = Quest("test", init, run)
        async with q:
            await q.get()  # ensure init finished
            assert q.get_nowait() == 99

    @pytest.mark.asyncio
    async def test_init_exception_propagates_via_get(self):
        async def init():
            raise ValueError("bad init")

        async def run(_):
            pass

        q = Quest("test", init, run)
        async with q:
            with pytest.raises(ValueError, match="bad init"):
                await q.get()

    @pytest.mark.asyncio
    async def test_close_not_called_when_init_fails(self):
        close_called = False

        async def init():
            raise RuntimeError("fail")

        async def run(_):
            pass

        async def close(_):
            nonlocal close_called
            close_called = True

        q = Quest("test", init, run, close)
        async with q:
            await asyncio.sleep(0.05)

        assert not close_called

    @pytest.mark.asyncio
    async def test_close_is_none(self):
        """Quest without a close callback still works."""
        log = []

        async def init():
            return "x"

        async def run(data):
            log.append(data)

        q = Quest("test", init, run, close=None)
        async with q:
            await asyncio.sleep(0.05)

        assert log == ["x"]

    @pytest.mark.asyncio
    async def test_from_run_step(self):
        ran = False

        async def my_run():
            nonlocal ran
            ran = True

        q = Quest.from_run_step("simple", my_run)
        async with q:
            await asyncio.sleep(0.05)

        assert ran

    @pytest.mark.asyncio
    async def test_remove_cancels_task(self):
        running = asyncio.Event()

        async def init():
            return None

        async def run(_):
            running.set()
            await asyncio.sleep(100)

        q = Quest("test", init, run)
        async with q:
            await running.wait()
            # task should be running
            assert not q.task.done()
        # After context exit, task is cancelled (need a tick for cancellation to propagate)
        await asyncio.sleep(0)
        assert q.task.cancelled()


# ---------------------------------------------------------------------------
# QuestManager
# ---------------------------------------------------------------------------


class TestQuestManager:
    @pytest.mark.asyncio
    async def test_add_and_run(self):
        log = []

        async def init():
            return 1

        async def run(d):
            log.append(d)

        async with QuestManager() as mgr:
            await mgr.add(Quest("q1", init, run))
            await asyncio.sleep(0.05)

        assert log == [1]

    @pytest.mark.asyncio
    async def test_add_replaces_same_name(self):
        """Adding a quest with the same name cancels the old one."""
        first_cancelled = asyncio.Event()

        async def init1():
            return "first"

        async def run1(_):
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                first_cancelled.set()
                raise

        async def init2():
            return "second"

        results = []

        async def run2(d):
            results.append(d)

        async with QuestManager() as mgr:
            await mgr.add(Quest("q", init1, run1))
            await asyncio.sleep(0.01)
            await mgr.add(Quest("q", init2, run2))
            await asyncio.sleep(0.05)

        assert first_cancelled.is_set()
        assert results == ["second"]

    @pytest.mark.asyncio
    async def test_remove_by_name(self):
        cancelled = False

        async def init():
            return None

        async def run(_):
            nonlocal cancelled
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cancelled = True
                raise

        async with QuestManager() as mgr:
            await mgr.add(Quest("q", init, run))
            await asyncio.sleep(0.01)
            await mgr.remove("q")
            await asyncio.sleep(0)  # let cancellation propagate

        assert cancelled

    @pytest.mark.asyncio
    async def test_remove_nonexistent_is_noop(self):
        async with QuestManager() as mgr:
            await mgr.remove("does-not-exist")  # should not raise

    @pytest.mark.asyncio
    async def test_wait_propagates_quest_exception(self):
        async def init():
            return None

        async def run(_):
            raise RuntimeError("boom")

        async with QuestManager() as mgr:
            await mgr.add(Quest("q", init, run))
            with pytest.raises(RuntimeError, match="boom"):
                await mgr.wait()

    @pytest.mark.asyncio
    async def test_exit_cleans_up_all_quests(self):
        tasks = []

        async def init():
            return None

        async def run(_):
            await asyncio.sleep(100)

        async with QuestManager() as mgr:
            for i in range(3):
                q = await mgr.add(Quest(f"q{i}", init, run))
                tasks.append(q.task)

        # All tasks should be cancelled after exiting (need a tick for cancellation to propagate)
        await asyncio.sleep(0)
        for t in tasks:
            assert t.cancelled()

    @pytest.mark.asyncio
    async def test_exit_tolerates_service_exceptions(self):
        """MissingServiceAtCapacity/Timeout/WebSocketClosedError are swallowed on exit."""
        for exc_cls in (MissingServiceAtCapacity, MissingServiceTimeout, WebSocketClosedError):

            async def init():
                return "data"

            async def run(_):
                await asyncio.sleep(100)

            async def make_close(exc_class):
                async def close(_):
                    if exc_class == MissingServiceAtCapacity:
                        raise exc_class("svc")
                    elif exc_class == MissingServiceTimeout:
                        raise exc_class("svc")
                    else:
                        raise exc_class()

                return close

            close_fn = await make_close(exc_cls)

            # Should not raise
            async with QuestManager() as mgr:
                await mgr.add(Quest("q", init, run, close_fn))
                await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_exit_logs_unknown_exceptions_in_close(self, caplog):
        """Non-service exceptions during close are logged, not raised."""

        async def init():
            return "data"

        async def run(_):
            await asyncio.sleep(100)

        async def close(_):
            raise TypeError("unexpected")

        async with QuestManager() as mgr:
            await mgr.add(Quest("q", init, run, close))
            await asyncio.sleep(0.01)

        assert "Error shutting down quest q" in caplog.text

    @pytest.mark.asyncio
    async def test_concurrent_quests(self):
        """Multiple quests with different names run concurrently."""
        results = []
        barrier = asyncio.Barrier(3)

        async def init():
            return None

        async def make_run(idx):
            async def run(_):
                await barrier.wait()
                results.append(idx)

            return run

        async with QuestManager() as mgr:
            for i in range(3):
                run_fn = await make_run(i)
                await mgr.add(Quest(f"q{i}", init, run_fn))
            await asyncio.sleep(0.1)

        assert sorted(results) == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_future_resolved_after_clean_exit(self):
        async def init():
            return None

        async def run(_):
            pass

        async with QuestManager() as mgr:
            await mgr.add(Quest("q", init, run))
            await asyncio.sleep(0.05)

        # _future should be resolved with None
        assert mgr._future.done()
        assert mgr._future.result() is None
