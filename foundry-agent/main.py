import asyncio
import os
from pathlib import Path

from runtime_mode import bare_metal_enabled, disable_optional_runtime_work


BARE_METAL_MODE = bare_metal_enabled()
if BARE_METAL_MODE:
    disable_optional_runtime_work()

from agent_framework import Agent, RawAgent
from agent_framework.foundry import FoundryChatClient, RawFoundryChatClient
from agent_framework.observability import disable_instrumentation
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential

from bare_metal import BARE_METAL_INSTRUCTIONS
from caller_identity import (
    elle_identity_status,
    validate_identity_binding_probe_nonce,
)
from continuity import load_continuity_config, make_continuity_tool
from private_tools import make_private_tools
from turn_memory import AutomaticTurnMemory
from wisdom_tools import make_wisdom_tools


def load_instructions() -> str:
    path = Path(__file__).with_name("instructions.txt")
    instructions = path.read_text(encoding="utf-8").strip()
    if not instructions:
        raise RuntimeError("instructions.txt is empty")
    return instructions


def build_agent(
    *,
    client,
    credential,
    name: str,
    instructions: str,
    bare_metal_mode: bool = BARE_METAL_MODE,
):
    if bare_metal_mode:
        return RawAgent(
            name=name,
            client=client,
            instructions=BARE_METAL_INSTRUCTIONS,
            middleware=[
                AutomaticTurnMemory(
                    endpoint=os.environ.get("ELLE_PRIVATE_TOOLS_ENDPOINT"),
                    client=client,
                )
            ],
            default_options={"store": False, "tools": []},
        )
    validate_identity_binding_probe_nonce(
        os.environ.get("ELLE_IDENTITY_BINDING_PROBE_NONCE")
    )
    continuity_config = load_continuity_config(
        os.environ.get("ELLE_CONTINUITY_ENDPOINT"),
        os.environ.get("ELLE_CONTINUITY_SCOPE"),
    )
    local_tools = [
        elle_identity_status,
        *make_private_tools(endpoint=os.environ.get("ELLE_PRIVATE_TOOLS_ENDPOINT")),
        *make_wisdom_tools(
            credential=credential,
            endpoint=os.environ.get("ELLE_WISDOM_TOOLS_ENDPOINT"),
            scope=os.environ.get("ELLE_WISDOM_SCOPE"),
        ),
    ]
    if continuity_config is not None:
        local_tools.append(
            make_continuity_tool(credential=credential, config=continuity_config)
        )
    options = {
        "name": name,
        "client": client,
        "instructions": instructions,
        "default_options": {"store": False},
        "middleware": [
            AutomaticTurnMemory(
                endpoint=os.environ.get("ELLE_PRIVATE_TOOLS_ENDPOINT"),
                client=client,
            )
        ],
    }
    return Agent(**options, tools=local_tools)


async def main() -> None:
    model = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5.6-luna")
    if BARE_METAL_MODE:
        disable_instrumentation()
    credential = DefaultAzureCredential(
        exclude_cli_credential=True,
        exclude_developer_cli_credential=True,
        exclude_interactive_browser_credential=True,
        exclude_powershell_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_visual_studio_code_credential=True,
    )

    client_type = RawFoundryChatClient if BARE_METAL_MODE else FoundryChatClient
    client = client_type(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=model,
        credential=credential,
    )
    agent = build_agent(
        name=os.environ.get("FOUNDRY_AGENT_NAME", "elle"),
        client=client,
        instructions=load_instructions(),
        credential=credential,
        bare_metal_mode=BARE_METAL_MODE,
    )

    host_options = (
        {"configure_observability": None, "access_log": None}
        if BARE_METAL_MODE
        else {}
    )
    server = ResponsesHostServer(agent, **host_options)
    await server.run_async()


if __name__ == "__main__":
    asyncio.run(main())
