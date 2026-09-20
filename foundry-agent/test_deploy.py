import contextlib
import io
import json
import sys
import unittest
import zipfile
from unittest.mock import MagicMock, patch

import deploy
from azure.ai.projects.models import (
    ActivityProtocolConfiguration,
    AgentEndpointConfig,
    BotServiceTenantAuthorizationScheme,
    EntraAuthorizationScheme,
    FixedRatioVersionSelectionRule,
    ProtocolConfiguration,
    ResponsesProtocolConfiguration,
    VersionSelector,
)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.project = MagicMock()
        self.project.agents.get_version.return_value.status = "active"
        self.project.agents.get.return_value.agent_endpoint = AgentEndpointConfig(
            version_selector=VersionSelector(
                version_selection_rules=[
                    FixedRatioVersionSelectionRule(agent_version="12", traffic_percentage=100)
                ]
            ),
            protocol_configuration=ProtocolConfiguration(
                responses=ResponsesProtocolConfiguration(),
                activity=ActivityProtocolConfiguration(),
            ),
            authorization_schemes=[
                EntraAuthorizationScheme(),
                BotServiceTenantAuthorizationScheme(),
            ],
        )

    def test_staging_requires_pinned_live_routing(self):
        self.project.agents.get.return_value.agent_endpoint = None
        with self.assertRaisesRegex(RuntimeError, "Pin live traffic"):
            deploy.deploy(self.project)
        self.project.agents.create_version_from_code.assert_not_called()

    def test_staging_packages_direct_private_tools(self):
        self.project.agents.create_version_from_code.return_value.version = "14"
        with patch.object(deploy, "wait_until_active"), patch.object(
            deploy, "package_source", return_value=(b"zip", "digest")
        ):
            self.assertEqual(deploy.deploy(self.project), "14")

        definition = self.project.agents.create_version_from_code.call_args.kwargs[
            "definition"
        ]
        self.assertEqual(
            definition.environment_variables,
            {
                "AZURE_AI_MODEL_DEPLOYMENT_NAME": "model-router",
                "ELLE_PRIVATE_TOOLS_ENDPOINT": deploy.PRIVATE_TOOLS_ENDPOINT,
                "ELLE_PRIVATE_TOOLS_SCOPE": deploy.PRIVATE_TOOLS_SCOPE,
                "ELLE_WISDOM_TOOLS_ENDPOINT": deploy.WISDOM_TOOLS_ENDPOINT,
                "ELLE_WISDOM_SCOPE": deploy.WISDOM_SCOPE,
            },
        )
        self.assertNotIn("TOOLBOX_ENDPOINT", definition.environment_variables)
        self.assertNotIn("ELLE_TOOLBOX_LIFETIME", definition.environment_variables)

    def test_source_package_includes_automatic_turn_memory(self):
        payload, _digest = deploy.package_source()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIn("turn_memory.py", archive.namelist())
            self.assertIn("bare_metal.py", archive.namelist())
            self.assertIn("runtime_mode.py", archive.namelist())
            self.assertIn("wisdom_tools.py", archive.namelist())
            instructions = archive.read("instructions.txt").decode("utf-8")

        self.assertIn("30% positive, 40% neutral and 30% negative", instructions)
        self.assertIn(
            "After every reply, the completed turn is appended",
            instructions,
        )
        self.assertIn(
            "checked for a meaningful diary reflection, relationship change",
            instructions,
        )
        self.assertIn("and newly attained durable knowledge", instructions)
        self.assertIn("Use `elle_shared_wisdom`", instructions)

    def test_staging_honors_model_override(self):
        self.project.agents.create_version_from_code.return_value.version = "14"
        with patch.dict(
            "os.environ", {"AZURE_AI_MODEL_DEPLOYMENT_NAME": "model-router"}
        ), patch.object(deploy, "wait_until_active"), patch.object(
            deploy, "package_source", return_value=(b"zip", "digest")
        ):
            deploy.deploy(self.project)

        definition = self.project.agents.create_version_from_code.call_args.kwargs[
            "definition"
        ]
        self.assertEqual(
            definition.environment_variables["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            "model-router",
        )

    def test_bare_metal_staging_disables_optional_runtime_work(self):
        self.project.agents.create_version_from_code.return_value.version = "19"
        with patch.object(deploy, "wait_until_active"), patch.object(
            deploy, "package_source", return_value=(b"zip", "digest")
        ):
            self.assertEqual(
                deploy.deploy(self.project, bare_metal_mode=True),
                "19",
            )

        definition = self.project.agents.create_version_from_code.call_args.kwargs[
            "definition"
        ]
        self.assertEqual(definition.environment_variables["ELLE_BARE_METAL_MODE"], "true")
        self.assertEqual(definition.environment_variables["ENABLE_INSTRUMENTATION"], "false")
        self.assertEqual(definition.environment_variables["OTEL_SDK_DISABLED"], "true")

    def test_cli_stages_without_catalog_options(self):
        project = MagicMock()
        client = MagicMock()
        client.return_value.__enter__.return_value = project
        output = io.StringIO()
        with patch.object(sys, "argv", ["deploy.py", "--identity-binding-probe-nonce", "demo"]), patch.object(
            deploy, "AzureCliCredential"
        ), patch.object(deploy, "AIProjectClient", client), patch.object(
            deploy, "deploy", return_value="14"
        ) as stage, contextlib.redirect_stdout(output):
            deploy.main()

        stage.assert_called_once_with(
            project,
            "demo",
            bare_metal_mode=False,
        )
        result = json.loads(output.getvalue())
        self.assertEqual(result["agentVersion"], "14")
        self.assertNotIn("tool_catalog", result)

    def test_cli_stages_full_memory_without_diagnostic_nonce(self):
        project = MagicMock()
        client = MagicMock()
        client.return_value.__enter__.return_value = project
        output = io.StringIO()
        with patch.object(sys, "argv", ["deploy.py", "--full-memory"]), patch.object(
            deploy, "AzureCliCredential"
        ), patch.object(deploy, "AIProjectClient", client), patch.object(
            deploy, "deploy", return_value="20"
        ) as stage, contextlib.redirect_stdout(output):
            deploy.main()

        stage.assert_called_once_with(
            project,
            None,
            bare_metal_mode=False,
        )
        result = json.loads(output.getvalue())
        self.assertEqual(result["agentVersion"], "20")
        self.assertFalse(result["bareMetalMode"])

    def test_promotion_keeps_existing_protocol_and_auth(self):
        deploy.promote(self.project, "14", "12")
        config = self.project.agents.update_details.call_args.kwargs["agent_endpoint"]
        self.assertEqual(
            config.version_selector.version_selection_rules[0].agent_version, "14"
        )
        self.assertEqual(config.authorization_schemes, self.project.agents.get.return_value.agent_endpoint.authorization_schemes)


if __name__ == "__main__":
    unittest.main()
