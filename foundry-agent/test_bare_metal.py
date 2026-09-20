import json
import os
import unittest
from unittest.mock import MagicMock, patch

from agent_framework import Agent, AgentSession, Message, RawAgent, SessionContext
from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    reset_request_context,
    set_request_context,
)

import main
from bare_metal import BARE_METAL_INSTRUCTIONS, BareMetalContextProvider
from runtime_mode import bare_metal_enabled, disable_optional_runtime_work


class BareMetalModeTests(unittest.IsolatedAsyncioTestCase):
    def test_flag_is_strict_and_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(bare_metal_enabled())
        for value in ("true", "TRUE", "1"):
            self.assertTrue(bare_metal_enabled(value))
        for value in ("false", "FALSE", "0", ""):
            self.assertFalse(bare_metal_enabled(value))
        with self.assertRaisesRegex(ValueError, "ELLE_BARE_METAL_MODE"):
            bare_metal_enabled("yes")

    def test_optional_runtime_work_is_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            disable_optional_runtime_work()
            self.assertEqual(os.environ["AGENT_FRAMEWORK_USER_AGENT_DISABLED"], "true")
            self.assertEqual(os.environ["AGENT_FRAMEWORK_FEATURE_MASK_DISABLED"], "true")
            self.assertEqual(os.environ["ENABLE_INSTRUMENTATION"], "false")
            self.assertEqual(os.environ["OTEL_SDK_DISABLED"], "true")

    def test_bare_agent_has_no_tools_and_uses_cognitive_lifecycle(self):
        agent = main.build_agent(
            client=MagicMock(),
            credential=MagicMock(),
            name="elle",
            instructions="full instructions",
            bare_metal_mode=True,
        )

        self.assertIsInstance(agent, RawAgent)
        self.assertNotIsInstance(agent, Agent)
        self.assertEqual(
            agent.default_options["instructions"],
            BARE_METAL_INSTRUCTIONS,
        )
        self.assertFalse(agent.default_options["store"])
        self.assertEqual(agent.default_options["tools"], [])
        self.assertEqual(len(agent.middleware), 1)
        self.assertEqual(type(agent.middleware[0]).__name__, "AutomaticTurnMemory")
        self.assertEqual(agent.context_providers, [])

    async def test_provider_prefetches_once_with_explicit_user(self):
        calls = []

        def load_context(**kwargs):
            calls.append(kwargs)
            return {
                "items": [{
                    "id": "static metadata must not be injected",
                    "occurred_at": "2026-09-15T10:00:00Z",
                    "text": f"dynamic {kwargs['store']} text",
                }],
            }

        provider = BareMetalContextProvider(
            endpoint="https://example.test/bridge",
            load_context=load_context,
        )
        context = SessionContext(
            input_messages=[Message("user", ["latest request"])],
        )
        request_context = set_request_context(
            FoundryAgentRequestContext(user_id="user-one")
        )
        try:
            await provider.before_run(
                agent=MagicMock(),
                session=AgentSession(),
                context=context,
                state={},
            )
        finally:
            reset_request_context(request_context)

        self.assertEqual({call["store"] for call in calls}, {"data_lake", "knowledge_base"})
        knowledge_call = next(call for call in calls if call["store"] == "knowledge_base")
        self.assertEqual(knowledge_call["query"], "latest request")
        self.assertEqual(knowledge_call["top"], 12)
        messages = context.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(
            json.loads(messages[0].text.removeprefix("PRIVATE_MEMORY_JSON:\n")),
            ["dynamic data_lake text", "dynamic knowledge_base text"],
        )
        self.assertNotIn("static metadata", messages[0].text)

    async def test_provider_injects_nothing_when_recall_is_empty(self):
        provider = BareMetalContextProvider(
            endpoint=None,
            load_context=lambda **_kwargs: {"items": []},
        )
        context = SessionContext(input_messages=[Message("user", ["hello"])])
        request_context = set_request_context(
            FoundryAgentRequestContext(user_id="user-one")
        )
        try:
            await provider.before_run(
                agent=MagicMock(),
                session=AgentSession(),
                context=context,
                state={},
            )
        finally:
            reset_request_context(request_context)

        self.assertEqual(context.get_messages(), [])

    async def test_provider_requires_platform_identity(self):
        provider = BareMetalContextProvider(
            endpoint=None,
            load_context=lambda **_kwargs: self.fail("context should not load"),
        )
        context = SessionContext(input_messages=[Message("user", ["hello"])])
        with self.assertRaisesRegex(RuntimeError, "platform identity"):
            await provider.before_run(
                agent=MagicMock(),
                session=AgentSession(),
                context=context,
                state={},
            )


if __name__ == "__main__":
    unittest.main()