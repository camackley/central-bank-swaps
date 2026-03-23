"""Tests for ClaudeCodeChatModel — subprocess-backed LangChain model."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from cbs.llm.claude_code_model import ClaudeCodeChatModel, ClaudeRateLimitError


def _make_completed_process(stdout: str = "", returncode: int = 0, stderr: str = ""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


class TestFormatMessages:
    def test_human_message_becomes_human_turn(self) -> None:
        model = ClaudeCodeChatModel()
        system, prompt = model._format_messages([HumanMessage(content="hello")])
        assert system == ""
        assert "Human: hello" in prompt

    def test_system_message_extracted(self) -> None:
        model = ClaudeCodeChatModel()
        system, prompt = model._format_messages(
            [SystemMessage(content="be helpful"), HumanMessage(content="hi")]
        )
        assert system == "be helpful"
        assert "Human: hi" in prompt

    def test_multiple_system_messages_joined(self) -> None:
        model = ClaudeCodeChatModel()
        system, _ = model._format_messages(
            [SystemMessage(content="part1"), SystemMessage(content="part2")]
        )
        assert "part1" in system
        assert "part2" in system


class TestCallCli:
    def test_basic_invocation_passes_prompt_via_stdin(self) -> None:
        model = ClaudeCodeChatModel()
        mock_result = _make_completed_process(stdout="ok")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = model._call_cli(system="", prompt="hello")
        assert result == "ok"
        cmd = mock_run.call_args[0][0]
        assert "-p" in cmd
        assert "--no-session-persistence" in cmd
        # Prompt is passed via stdin, not as a CLI argument
        assert "hello" not in cmd
        assert mock_run.call_args.kwargs["input"] == "hello"

    def test_system_prompt_included_when_set(self) -> None:
        model = ClaudeCodeChatModel()
        mock_result = _make_completed_process(stdout="ok")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            model._call_cli(system="be concise", prompt="hello")
        cmd = mock_run.call_args[0][0]
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert cmd[idx + 1] == "be concise"

    def test_system_prompt_omitted_when_empty(self) -> None:
        model = ClaudeCodeChatModel()
        mock_result = _make_completed_process(stdout="ok")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            model._call_cli(system="", prompt="hello")
        cmd = mock_run.call_args[0][0]
        assert "--system-prompt" not in cmd

    def test_json_schema_flag_included(self) -> None:
        model = ClaudeCodeChatModel()
        mock_result = _make_completed_process(stdout='{"value": 1}')
        schema = '{"type": "object"}'
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            model._call_cli(system="", prompt="hello", json_schema=schema)
        cmd = mock_run.call_args[0][0]
        assert "--json-schema" in cmd
        idx = cmd.index("--json-schema")
        assert cmd[idx + 1] == schema

    def test_runtime_error_on_nonzero_exit(self) -> None:
        model = ClaudeCodeChatModel()
        mock_result = _make_completed_process(returncode=1, stderr="some error")
        with (
            patch("subprocess.run", return_value=mock_result),
            pytest.raises(RuntimeError, match="some error"),
        ):
            model._call_cli(system="", prompt="hello")

    def test_rate_limit_raises_claude_rate_limit_error(self) -> None:
        model = ClaudeCodeChatModel()
        mock_result = _make_completed_process(
            returncode=1, stderr="Error: usage limit exceeded for today"
        )
        with (
            patch("subprocess.run", return_value=mock_result),
            pytest.raises(ClaudeRateLimitError, match="usage limit"),
        ):
            model._call_cli(system="", prompt="hello")

    @pytest.mark.parametrize(
        "stderr_msg",
        [
            "rate limit reached",
            "too many requests",
            "daily limit exceeded",
            "monthly limit reached",
        ],
    )
    def test_rate_limit_patterns_detected(self, stderr_msg: str) -> None:
        model = ClaudeCodeChatModel()
        mock_result = _make_completed_process(returncode=1, stderr=stderr_msg)
        with (
            patch("subprocess.run", return_value=mock_result),
            pytest.raises(ClaudeRateLimitError),
        ):
            model._call_cli(system="", prompt="hello")

    def test_allowed_tools_flag_included(self) -> None:
        model = ClaudeCodeChatModel()
        mock_result = _make_completed_process(stdout="ok")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            model._call_cli(
                system="",
                prompt="hello",
                allowed_tools=[
                    "mcp__playwright__browser_navigate",
                    "mcp__playwright__browser_snapshot",
                ],
            )
        cmd = mock_run.call_args[0][0]
        assert "--allowedTools" in cmd
        idx = cmd.index("--allowedTools")
        assert (
            cmd[idx + 1]
            == "mcp__playwright__browser_navigate,mcp__playwright__browser_snapshot"
        )

    def test_timeout_override(self) -> None:
        model = ClaudeCodeChatModel(timeout=60)
        mock_result = _make_completed_process(stdout="ok")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            model._call_cli(system="", prompt="hello", timeout_override=300)
        assert mock_run.call_args.kwargs["timeout"] == 300

    def test_model_flag_uses_model_name(self) -> None:
        model = ClaudeCodeChatModel(model_name="claude-haiku-4-5")
        mock_result = _make_completed_process(stdout="ok")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            model._call_cli(system="", prompt="hello")
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-haiku-4-5"


class TestGenerate:
    def test_returns_ai_message_content(self) -> None:
        model = ClaudeCodeChatModel()
        mock_result = _make_completed_process(stdout="the answer is 42")
        with patch("subprocess.run", return_value=mock_result):
            result = model.invoke([HumanMessage(content="what is 6*7?")])
        assert isinstance(result, AIMessage)
        assert result.content == "the answer is 42"


class TestWithStructuredOutput:
    def test_returns_validated_pydantic_model(self) -> None:
        class MySchema(BaseModel):
            value: int
            label: str

        model = ClaudeCodeChatModel()
        raw_json = json.dumps({"value": 42, "label": "hello"})
        mock_result = _make_completed_process(stdout=raw_json)
        with patch("subprocess.run", return_value=mock_result):
            structured = model.with_structured_output(MySchema)
            result = structured.invoke([HumanMessage(content="give me data")])

        assert isinstance(result, MySchema)
        assert result.value == 42
        assert result.label == "hello"

    def test_schema_injected_in_system_prompt(self) -> None:
        class MySchema(BaseModel):
            name: str

        model = ClaudeCodeChatModel()
        raw_json = json.dumps({"name": "test"})
        mock_result = _make_completed_process(stdout=raw_json)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            structured = model.with_structured_output(MySchema)
            structured.invoke([HumanMessage(content="hi")])

        cmd = mock_run.call_args[0][0]
        # Schema is injected into the system prompt, not as a CLI flag
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        system_prompt = cmd[idx + 1]
        assert "MySchema" in system_prompt or "properties" in system_prompt
