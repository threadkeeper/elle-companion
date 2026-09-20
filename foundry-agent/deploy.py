import argparse
import copy
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AgentEndpointConfig,
    CodeConfiguration,
    CodeDependencyResolution,
    FixedRatioVersionSelectionRule,
    HostedAgentDefinition,
    ProtocolVersionRecord,
    VersionRefIndicator,
    VersionSelector,
)
from azure.identity import AzureCliCredential


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent
PROJECT_ENDPOINT = (
    "https://foundry-jva-002.services.ai.azure.com/api/projects/"
    "proj-default-sweden"
)
AGENT_NAME = "elle"
DEFAULT_MODEL_DEPLOYMENT_NAME = "gpt-5.6-luna"
PRIVATE_TOOLS_ENDPOINT = (
    "https://elle-private-vnet.yellowsky-9d92d540.swedencentral."
    "azurecontainerapps.io/bridge"
)
WISDOM_TOOLS_ENDPOINT = (
    "https://elle-wisdom-vnet.yellowsky-9d92d540.swedencentral."
    "azurecontainerapps.io/bridge"
)
WISDOM_SCOPE = "api://0479a728-6b4d-4d96-8693-ef766bc8e1fe/.default"
PRIVATE_TOOLS_SCOPE = "api://0479a728-6b4d-4d96-8693-ef766bc8e1fe/.default"
CONTINUITY_ENDPOINT = (
    "https://elle-private-vnet.yellowsky-9d92d540.swedencentral."
    "azurecontainerapps.io/continuity/context"
)
CONTINUITY_SCOPE = "api://0479a728-6b4d-4d96-8693-ef766bc8e1fe/.default"
_PROBE_NONCE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")


def model_deployment_name() -> str:
    return os.environ.get(
        "AZURE_AI_MODEL_DEPLOYMENT_NAME", DEFAULT_MODEL_DEPLOYMENT_NAME
    )


def prompt_text() -> str:
    text = (ROOT / "ELLE_AGENT_PROMPT.md").read_text(encoding="utf-8")
    if text.startswith("---"):
        _, _, text = text.partition("---")
        _, separator, text = text.partition("---")
        if not separator:
            raise RuntimeError("Elle prompt front matter is incomplete")
    return text.strip() + "\n"


def package_source() -> tuple[bytes, str]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(SOURCE / "main.py", "main.py")
        archive.write(SOURCE / "caller_identity.py", "caller_identity.py")
        archive.write(SOURCE / "bare_metal.py", "bare_metal.py")
        archive.write(SOURCE / "continuity.py", "continuity.py")
        archive.write(SOURCE / "private_tools.py", "private_tools.py")
        archive.write(SOURCE / "runtime_mode.py", "runtime_mode.py")
        archive.write(SOURCE / "turn_memory.py", "turn_memory.py")
        archive.write(SOURCE / "wisdom_tools.py", "wisdom_tools.py")
        archive.write(SOURCE / "requirements.txt", "requirements.txt")
        archive.writestr("instructions.txt", prompt_text())
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def wait_until_active(
    project: AIProjectClient, version: str, timeout_seconds: int = 900
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        details = project.agents.get_version(AGENT_NAME, version)
        status = details["status"]
        print(f"Elle {version}: {status}", flush=True)
        if status == "active":
            return
        if status == "failed":
            raise RuntimeError(
                f"Hosted agent provisioning failed: {dict(details)}"
            )
        time.sleep(10)
    raise TimeoutError("Timed out waiting for the hosted agent to become active")


def deploy(
    project: AIProjectClient,
    identity_binding_probe_nonce: str | None = None,
    bare_metal_mode: bool = False,
) -> str:
    endpoint = project.agents.get(AGENT_NAME).agent_endpoint
    if endpoint is None or endpoint.version_selector is None:
        raise RuntimeError("Pin live traffic to a version before staging")
    rules = endpoint.version_selector.version_selection_rules
    if not rules or any(rule.type != "FixedRatio" for rule in rules):
        raise RuntimeError("Pin live traffic to a version before staging")
    code, digest = package_source()
    environment_variables = {
        "AZURE_AI_MODEL_DEPLOYMENT_NAME": model_deployment_name(),
        "ELLE_PRIVATE_TOOLS_ENDPOINT": PRIVATE_TOOLS_ENDPOINT,
        "ELLE_PRIVATE_TOOLS_SCOPE": PRIVATE_TOOLS_SCOPE,
        "ELLE_WISDOM_TOOLS_ENDPOINT": WISDOM_TOOLS_ENDPOINT,
        "ELLE_WISDOM_SCOPE": WISDOM_SCOPE,
    }
    if bare_metal_mode:
        environment_variables.update({
            "ELLE_BARE_METAL_MODE": "true",
            "AGENT_FRAMEWORK_USER_AGENT_DISABLED": "true",
            "AGENT_FRAMEWORK_FEATURE_MASK_DISABLED": "true",
            "ENABLE_INSTRUMENTATION": "false",
            "OTEL_SDK_DISABLED": "true",
        })
    if identity_binding_probe_nonce is not None:
        if not _PROBE_NONCE_PATTERN.fullmatch(identity_binding_probe_nonce):
            raise ValueError("Identity binding probe nonce must be safe bounded ASCII")
        environment_variables["ELLE_IDENTITY_BINDING_PROBE_NONCE"] = (
            identity_binding_probe_nonce
        )
    created = project.agents.create_version_from_code(
        agent_name=AGENT_NAME,
        description=(
            "Elle hosted identity diagnostic with governed enterprise tools."
        ),
        definition=HostedAgentDefinition(
            cpu="1",
            memory="2Gi",
            code_configuration=CodeConfiguration(
                runtime="python_3_13",
                entry_point=["python", "main.py"],
                dependency_resolution=CodeDependencyResolution.REMOTE_BUILD,
            ),
            environment_variables=environment_variables,
            protocol_versions=[
                ProtocolVersionRecord(protocol="responses", version="2.0.0")
            ],
        ),
        code=("elle.zip", code, "application/zip"),
        code_zip_sha256=digest,
    )
    wait_until_active(project, created.version)
    return created.version


def promote(project: AIProjectClient, version: str, expected_live_version: str) -> None:
    candidate = project.agents.get_version(AGENT_NAME, version)
    if candidate.status != "active":
        raise RuntimeError(f"Elle {version} is not active")
    current = project.agents.get(AGENT_NAME)
    endpoint = current.agent_endpoint
    if endpoint is None:
        raise RuntimeError("Elle has no existing endpoint configuration")
    rules = endpoint.version_selector.version_selection_rules
    if len(rules) != 1 or rules[0].agent_version != expected_live_version:
        raise RuntimeError("Live routing changed; inspect it before promotion")
    if rules[0].traffic_percentage != 100:
        raise RuntimeError("Refusing to overwrite split-traffic routing")
    config = AgentEndpointConfig(
        version_selector=VersionSelector(
            version_selection_rules=[
                FixedRatioVersionSelectionRule(
                    agent_version=version, traffic_percentage=100
                )
            ]
        ),
        protocol_configuration=copy.deepcopy(endpoint.protocol_configuration),
        authorization_schemes=copy.deepcopy(endpoint.authorization_schemes),
    )
    project.agents.update_details(AGENT_NAME, agent_endpoint=config)


def smoke_test(project: AIProjectClient, version: str) -> str:
    session = project.agents.create_session(
        AGENT_NAME, version_indicator=VersionRefIndicator(agent_version=version)
    )
    if not session.agent_session_id:
        raise RuntimeError("Foundry did not return a candidate session ID")
    try:
        with project.get_openai_client(agent_name=AGENT_NAME) as client:
            response = client.responses.create(
                input=(
                    "Read-only deployment diagnostic: do not save a memory. Use "
                    "Microsoft Learn to name the protocol used by Foundry hosted agents."
                ),
                extra_body={"session_id": session.agent_session_id},
            )
    finally:
        project.agents.stop_session(AGENT_NAME, session.agent_session_id)
    if not response.output_text.strip():
        raise RuntimeError("Elle returned an empty smoke-test response")
    return response.output_text.strip()


def main() -> None:
    global PROJECT_ENDPOINT

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-endpoint", default=PROJECT_ENDPOINT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--promote-version", help="Promote an already-tested version")
    action.add_argument("--test-version", help="Smoke-test one explicit candidate")
    parser.add_argument("--expected-live-version", help="Required guard for promotion")
    parser.add_argument(
        "--identity-binding-probe-nonce",
        help="Stage a separate temporary identity binding probe candidate",
    )
    parser.add_argument(
        "--bare-metal",
        action="store_true",
        help="Stage an unpromoted one-model-call benchmark candidate",
    )
    parser.add_argument(
        "--full-memory",
        action="store_true",
        help="Stage an unpromoted candidate with private memory actions",
    )
    args = parser.parse_args()
    if args.promote_version and not args.expected_live_version:
        parser.error("--promote-version requires --expected-live-version")
    if (args.promote_version or args.test_version) and (
        args.identity_binding_probe_nonce or args.bare_metal or args.full_memory
    ):
        parser.error("Testing/promotion cannot be combined with staging options")
    if args.bare_metal and args.full_memory:
        parser.error("Choose either --bare-metal or --full-memory")
    if not (
        args.promote_version
        or args.test_version
        or args.identity_binding_probe_nonce is not None
        or args.bare_metal
        or args.full_memory
    ):
        parser.error("Staging requires an explicit action")

    PROJECT_ENDPOINT = args.project_endpoint.rstrip("/")
    credential = AzureCliCredential(process_timeout=120)
    with AIProjectClient(PROJECT_ENDPOINT, credential, allow_preview=True) as project:
        if args.promote_version:
            promote(project, args.promote_version, args.expected_live_version)
            print(json.dumps({"promotedVersion": args.promote_version}))
            return
        if args.test_version:
            print(smoke_test(project, args.test_version))
            return
        agent_version = deploy(
            project,
            args.identity_binding_probe_nonce,
            bare_metal_mode=args.bare_metal,
        )

    print(
        json.dumps(
            {
                "agent": AGENT_NAME,
                "agentVersion": agent_version,
                "model": model_deployment_name(),
                "bareMetalMode": args.bare_metal,
                "projectEndpoint": PROJECT_ENDPOINT,
                "promoted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Deployment failed: {error}", file=sys.stderr)
        raise
