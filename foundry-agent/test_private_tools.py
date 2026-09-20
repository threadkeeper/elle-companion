import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    reset_request_context,
    set_request_context,
)

import private_tools


class Response:
    status = 200

    def read(self, _limit):
        return b'{"ok":true}'

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class PrivateToolsTests(unittest.TestCase):
    def setUp(self):
        self.credential = MagicMock()
        self.credential.get_token.return_value = SimpleNamespace(token="workload-token")
        self.scope = "api://elle/.default"

    def test_automatic_cognitive_helpers_use_explicit_user_binding(self):
        captured = []

        def open_url(request, timeout):
            captured.append((request, timeout))
            return Response()

        with patch("private_tools.urllib.request.urlopen", side_effect=open_url):
            private_tools.query_cognitive_store(
                endpoint="https://example.test/bridge",
                credential=self.credential,
                scope=self.scope,
                user_id="explicit-user",
                store="knowledge_base",
                query="fast recall",
                top=12,
            )
            private_tools.save_cognitive_knowledge(
                endpoint="https://example.test/bridge",
                credential=self.credential,
                scope=self.scope,
                user_id="explicit-user",
                content="The user prefers concise answers.",
                salience=0.8,
                timestamp="2026-09-15T12:00:00Z",
            )

        query_request, save_request = (item[0] for item in captured)
        self.assertEqual(
            query_request.get_header("Authorization"),
            "Bearer workload-token",
        )
        self.assertIsNone(
            query_request.get_header("X-elle-continuity-handle-sha256")
        )
        self.assertEqual(query_request.full_url, "https://example.test/bridge/elle_cognitive_query")
        self.assertEqual(json.loads(query_request.data), {
            "store": "knowledge_base", "mode": "auto", "query": "fast recall",
            "from": None, "to": None, "order": "newest", "top": 12,
            "count_only": False,
            "user_object_id": "explicit-user",
        })
        self.assertEqual(save_request.full_url, "https://example.test/bridge/elle_save_cognitive")
        self.assertEqual(json.loads(save_request.data), {
            "store": "knowledge_base",
            "timestamp": "2026-09-15T12:00:00Z",
            "content": "The user prefers concise answers.",
            "salience": 0.8,
            "user_object_id": "explicit-user",
        })

    def test_private_tools_expose_only_new_cognitive_memory_schema(self):
        names = [
            tool.__name__
            for tool in private_tools.make_private_tools(credential=self.credential)
        ]
        self.assertEqual(names, [
            "elle_personality",
            "elle_set_personality",
            "elle_cognitive_query",
            "elle_save_cognitive",
        ])
        for legacy in (
            "elle_context", "elle_list_memories", "elle_remember",
            "elle_correct", "elle_forget",
        ):
            self.assertNotIn(legacy, names)

    def test_current_user_binding_is_resolved_on_each_call(self):
        tool = private_tools.make_private_tools(credential=self.credential)[2]
        with patch("private_tools.urllib.request.urlopen", return_value=Response()) as open_url:
            for user_id in ("demo-one", "demo-two"):
                token = set_request_context(FoundryAgentRequestContext(user_id=user_id))
                try:
                    tool("knowledge_base", query="dashboard")
                    request = open_url.call_args.args[0]
                    self.assertEqual(
                        json.loads(request.data)["user_object_id"],
                        user_id,
                    )
                finally:
                    reset_request_context(token)

    def test_archive_turn_uses_dedicated_bounded_contract(self):
        with patch("private_tools.urllib.request.urlopen", return_value=Response()) as open_url:
            private_tools.archive_conversation_turn(
                endpoint="https://example.test/bridge",
                credential=self.credential,
                scope=self.scope,
                user_id="background-user",
                user_text="hello",
                assistant_text="reply",
                response_id="response-1",
            )

        request = open_url.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.test/bridge/elle_archive_turn")
        self.assertEqual(body["user_text"], "hello")
        self.assertEqual(body["assistant_text"], "reply")
        self.assertTrue(body["timestamp"].endswith("Z"))
        self.assertEqual(
            set(body),
            {"timestamp", "user_text", "assistant_text", "user_object_id"},
        )

    def test_cognitive_tools_send_only_bounded_structured_contracts(self):
        captured = []

        def open_url(request, timeout):
            captured.append((request.full_url, json.loads(request.data), timeout))
            return Response()

        tools = private_tools.make_private_tools(
            credential=self.credential,
            endpoint="https://example.test/bridge",
            scope=self.scope,
        )
        with patch("private_tools.urllib.request.urlopen", side_effect=open_url):
            token = set_request_context(FoundryAgentRequestContext(user_id="demo-user"))
            try:
                tools[-2]("knowledge_base", "semantic", "green dashboard", "oldest", "2026-01-01", "2026-12-31", 500, True)
                tools[-1]("diary", "A durable reflection")
                tools[-1]("knowledge_base", "Prefers green dashboards", 0.85)
            finally:
                reset_request_context(token)

        self.assertEqual(captured[0][0], "https://example.test/bridge/elle_cognitive_query")
        self.assertEqual(captured[0][1], {
            "store": "knowledge_base", "mode": "semantic", "query": "green dashboard",
            "from": "2026-01-01", "to": "2026-12-31", "order": "oldest",
            "top": 500, "count_only": True, "user_object_id": "demo-user",
        })
        self.assertEqual(captured[1][0], "https://example.test/bridge/elle_save_cognitive")
        self.assertEqual(captured[1][1]["store"], "diary")
        self.assertEqual(captured[1][1]["content"], "A durable reflection")
        self.assertIsNone(captured[1][1]["salience"])
        self.assertEqual(captured[2][1]["store"], "knowledge_base")
        self.assertEqual(captured[2][1]["salience"], 0.85)
        self.assertNotIn("owner_id", json.dumps(captured))
        self.assertNotIn("container", json.dumps(captured))
        self.assertNotIn("sql", json.dumps(captured).lower())


if __name__ == "__main__":
    unittest.main()
