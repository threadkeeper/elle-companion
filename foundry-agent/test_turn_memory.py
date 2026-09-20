import asyncio
import threading
import time
import unittest
from types import SimpleNamespace

from agent_framework import AgentContext, AgentResponse, Message
from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    reset_request_context,
    set_request_context,
)

from turn_memory import AutomaticTurnMemory


class AutomaticTurnMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_reply_does_not_wait_for_background_save(self):
        started = threading.Event()
        release = threading.Event()
        captured = []

        def save_turn(**kwargs):
            captured.append(kwargs)
            started.set()
            release.wait(2)

        middleware = AutomaticTurnMemory(
            endpoint="https://example.test/bridge",
            save_turn=save_turn,
            load_context=lambda **_kwargs: {"items": []},
        )
        context = AgentContext(
            agent=object(),
            messages=[Message("user", ["latest user turn"])],
            stream=True,
        )

        async def call_next():
            return None

        token = set_request_context(FoundryAgentRequestContext(user_id="demo-user"))
        try:
            await middleware.process(context, call_next)
        finally:
            reset_request_context(token)

        self.assertEqual(len(context.stream_result_hooks), 1)
        response = AgentResponse(
            messages=[Message("assistant", ["final Elle reply"])],
            response_id="response-1",
        )
        before = time.perf_counter()
        returned = context.stream_result_hooks[0](response)
        elapsed = time.perf_counter() - before

        self.assertIs(returned, response)
        self.assertLess(elapsed, 0.05)
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        self.assertEqual(len(middleware.pending), 1)
        self.assertEqual(
            captured,
            [{
                "endpoint": "https://example.test/bridge",
                "credential": None,
                "scope": None,
                "user_id": "demo-user",
                "user_text": "latest user turn",
                "assistant_text": "final Elle reply",
                "response_id": "response-1",
            }],
        )

        pending = list(middleware.pending)
        release.set()
        await asyncio.gather(*pending)
        self.assertFalse(middleware.pending)

    async def test_empty_completed_turn_is_not_queued(self):
        middleware = AutomaticTurnMemory(
            endpoint=None,
            save_turn=lambda **_kwargs: self.fail("save should not run"),
            load_context=lambda **_kwargs: {"items": []},
        )
        response = AgentResponse(messages=[Message("assistant", [""])])
        self.assertIs(middleware._queue(response, "demo-user", "hello"), response)
        self.assertFalse(middleware.pending)

    async def test_retrieval_is_injected_before_generation(self):
        calls = []

        def load_context(**kwargs):
            calls.append(kwargs)
            return {
                "items": [
                    {
                        "id": f"{kwargs['store']}-1",
                        "occurred_at": "2026-09-15T10:00:00Z",
                        "text": f"saved {kwargs['store']}",
                    }
                ]
            }

        middleware = AutomaticTurnMemory(
            endpoint="https://example.test/bridge",
            load_context=load_context,
        )
        context = AgentContext(
            agent=object(),
            messages=[Message("user", ["What did I decide?"])],
            stream=True,
        )

        async def call_next():
            self.assertEqual(context.messages[0].role, "system")
            self.assertIn("saved data_lake", context.messages[0].text)
            self.assertIn("saved knowledge_base", context.messages[0].text)

        token = set_request_context(FoundryAgentRequestContext(user_id="demo-user"))
        try:
            await middleware.process(context, call_next)
        finally:
            reset_request_context(token)

        self.assertEqual(
            {call["store"] for call in calls},
            {"data_lake", "knowledge_base", "diary", "connections"},
        )
        knowledge_call = next(call for call in calls if call["store"] == "knowledge_base")
        self.assertEqual(knowledge_call["query"], "What did I decide?")

    async def test_completed_turn_archives_then_saves_attained_knowledge(self):
        events = []

        class Client:
            async def get_response(self, messages, **kwargs):
                events.append(("assess", messages, kwargs))
                return SimpleNamespace(
                    value={
                        "facts": [
                            {"content": "The user prefers concise answers.", "salience": 0.8},
                            {"content": "", "salience": 0.4},
                        ],
                        "diary": None,
                        "connection": None,
                    }
                )

        middleware = AutomaticTurnMemory(
            endpoint="https://example.test/bridge",
            client=Client(),
            save_turn=lambda **kwargs: events.append(("archive", kwargs)),
            load_context=lambda **_kwargs: {"items": []},
            save_knowledge=lambda **kwargs: events.append(("knowledge", kwargs)),
        )
        await middleware._save(
            user_id="demo-user",
            user_text="I prefer concise answers.",
            assistant_text="Understood.",
            response_id="response-2",
        )

        self.assertEqual([event[0] for event in events], ["archive", "assess", "knowledge"])
        self.assertEqual(events[2][1]["content"], "The user prefers concise answers.")
        self.assertEqual(events[2][1]["salience"], 0.8)
        self.assertEqual(events[1][2]["options"]["tools"], [])
        self.assertFalse(events[1][2]["options"]["store"])
        schema = events[1][2]["options"]["response_format"]
        self.assertEqual(schema["required"], ["facts", "diary", "connection"])
        self.assertEqual(schema["properties"]["connection"]["type"], ["object", "null"])

    async def test_successful_archive_precedes_diary_connection_and_knowledge(self):
        events = []

        class Client:
            async def get_response(self, _messages, **_kwargs):
                events.append(("assess", {}))
                return SimpleNamespace(value={
                    "facts": [{"content": "A durable fact.", "salience": 0.7}],
                    "diary": {"content": "A meaningful reflection."},
                    "connection": {"change_amount": 0.2, "note": "Trust grew."},
                })

        middleware = AutomaticTurnMemory(
            endpoint=None,
            client=Client(),
            save_turn=lambda **kwargs: events.append(("archive", kwargs)),
            load_context=lambda **_kwargs: {"items": []},
            save_diary=lambda **kwargs: events.append(("diary", kwargs)),
            save_connection=lambda **kwargs: events.append(("connection", kwargs)),
            save_knowledge=lambda **kwargs: events.append(("knowledge", kwargs)),
            save_telemetry=lambda **kwargs: events.append(("telemetry", kwargs)),
        )
        await middleware._save(
            user_id="demo-user",
            user_text="I trust this process more now.",
            assistant_text="That matters.",
            response_id="response-order",
        )

        self.assertEqual(
            [event[0] for event in events],
            ["archive", "assess", "diary", "connection", "knowledge", "telemetry"],
        )
        self.assertEqual(events[3][1]["event_key"], "response-order")
        self.assertTrue(events[5][1]["persisted"])

    async def test_no_attained_knowledge_writes_only_the_daily_archive(self):
        events = []

        class Client:
            async def get_response(self, _messages, **_kwargs):
                events.append("assess")
                return SimpleNamespace(value={"facts": [], "diary": None, "connection": None})

        middleware = AutomaticTurnMemory(
            endpoint=None,
            client=Client(),
            save_turn=lambda **_kwargs: events.append("archive"),
            load_context=lambda **_kwargs: {"items": []},
            save_knowledge=lambda **_kwargs: self.fail("knowledge should not save"),
        )
        await middleware._save(
            user_id="demo-user",
            user_text="What time is it?",
            assistant_text="It is noon.",
            response_id="response-3",
        )

        self.assertEqual(events, ["archive", "assess"])


if __name__ == "__main__":
    unittest.main()
