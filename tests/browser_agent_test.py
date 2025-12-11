# -*- coding: utf-8 -*-
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from agentscope.message import Msg
from agentscope.tool import Toolkit
from agentscope.memory import MemoryBase
from agentscope.model import ChatModelBase
from agentscope.formatter import FormatterBase
from browser_use.agent_browser.browser_agent import BrowserAgent


@pytest.fixture
def mock_dependencies() -> Dict[str, MagicMock]:
    return {
        "model": MagicMock(spec=ChatModelBase),
        "formatter": MagicMock(spec=FormatterBase),
        "memory": MagicMock(spec=MemoryBase),
        "toolkit": MagicMock(spec=Toolkit),
    }


@pytest.fixture
def agent(
    # pylint: disable=redefined-outer-name
    mock_dependencies: Dict[str, MagicMock],
) -> BrowserAgent:
    return BrowserAgent(
        name="TestBot",
        model=mock_dependencies["model"],
        formatter=mock_dependencies["formatter"],
        memory=mock_dependencies["memory"],
        toolkit=mock_dependencies["toolkit"],
        start_url="https://test.com",
    )

def test_hooks_registered(
    agent: BrowserAgent,  # pylint: disable=redefined-outer-name
) -> None:
    """Verify instance-level hooks are registered"""
    # Disable pylint warning for protected member access
    assert hasattr(
        agent,
        "_instance_pre_reply_hooks",
    )  # pylint: disable=protected-access
    assert (
        "browser_agent_default_url_pre_reply"
        # pylint: disable=protected-access
        in agent._instance_pre_reply_hooks
    )

    assert hasattr(
        agent,
        "_instance_pre_reasoning_hooks",
    )  # pylint: disable=protected-access
    assert (
        "browser_agent_observe_pre_reasoning"
        # pylint: disable=protected-access
        in agent._instance_pre_reasoning_hooks
    )

@pytest.mark.asyncio
async def test_pre_reply_hook_navigation(
    agent: BrowserAgent,  # pylint: disable=redefined-outer-name
) -> None:
    # pylint: disable=protected-access
    agent._has_initial_navigated = False

    # Get instance-level hook function
    # pylint: disable=protected-access
    hook_func = agent._instance_pre_reply_hooks[
        "browser_agent_default_url_pre_reply"
    ]
    await hook_func(agent)  # Directly invoke hook function

    # pylint: disable=protected-access
    assert agent._has_initial_navigated is True
    assert agent.toolkit.call_tool_function.called


@pytest.mark.asyncio
async def test_observe_pre_reasoning(
    agent: BrowserAgent,  # pylint: disable=redefined-outer-name
) -> None:
    # Mock tool response (fix: use Msg object with content attribute)
    mock_response = AsyncMock()
    mock_response.__aiter__.return_value = [
        Msg("system", [{"text": "Snapshot content"}], "system"),
    ]
    agent.toolkit.call_tool_function = AsyncMock(
        return_value=mock_response,
    )

    # Replace memory add method
    with patch.object(
        agent.memory,
        "add",
        new_callable=AsyncMock,
    ) as mock_add:
        # Get instance-level hook function
        # pylint: disable=protected-access
        hook_func = agent._instance_pre_reasoning_hooks[
            "browser_agent_observe_pre_reasoning"
        ]
        await hook_func(agent)  # Directly invoke hook function

        mock_add.assert_awaited_once()
        added_msg = mock_add.call_args[0][0]
        assert "Snapshot content" in added_msg.content[0]["text"]


def test_filter_execution_text(
    agent: BrowserAgent,  # pylint: disable=redefined-outer-name
) -> None:
    text = """
    ### New console messages
    Some console output
    ###
    ### Page state
    YAML content here
    ```yaml
    key: value
    ```
    Regular text content
    """
    # pylint: disable=protected-access
    filtered = agent._filter_execution_text(text)

    assert "console output" not in filtered
    assert "key: value" not in filtered
    assert "Regular text content" in filtered
    assert "YAML content" in filtered
