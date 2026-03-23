"""ClaudeCodeChatModel — LangChain model backed by the ``claude`` CLI subprocess."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

_RATE_LIMIT_PATTERNS = (
    "rate limit",
    "too many requests",
    "usage limit",
    "daily limit",
    "monthly limit",
)


class ClaudeRateLimitError(Exception):
    """Raised when the claude CLI hits the subscription usage limit."""


class ClaudeCodeChatModel(BaseChatModel):
    """BaseChatModel that spawns ``claude -p`` for each invocation.

    Calls bill against the locally-authenticated Claude Code Max subscription
    instead of Anthropic API credits.
    """

    model_name: str = Field(default="claude-sonnet-4-6")
    claude_path: str = Field(default="claude")
    timeout: int = Field(default=120)

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    def _format_messages(self, messages: list[BaseMessage]) -> tuple[str, str]:
        """Return (system_prompt, conversation_prompt) from a message list."""
        system_parts: list[str] = []
        turns: list[str] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_parts.append(str(msg.content))
            else:
                role = "Human" if msg.type == "human" else "Assistant"
                turns.append(f"{role}: {msg.content}")
        return "\n\n".join(system_parts), "\n\n".join(turns)

    def _call_cli(
        self,
        system: str,
        prompt: str,
        json_schema: str | None = None,
        *,
        allowed_tools: list[str] | None = None,
        timeout_override: int | None = None,
    ) -> str:
        """Invoke claude CLI and return the raw text response.

        The prompt is passed via **stdin** (not as a CLI argument) to avoid
        OS ``[Errno 7] Argument list too long`` errors with large prompts.

        Args:
            system: System prompt text.
            prompt: User prompt text (piped via stdin).
            json_schema: Optional JSON schema for structured output.
            allowed_tools: Optional list of MCP tool names to allow
                (e.g. ``["mcp__playwright__browser_navigate"]``).
            timeout_override: Override the default subprocess timeout (seconds).
        """
        cmd = [
            self.claude_path,
            "-p",
            "--model",
            self.model_name,
            "--no-session-persistence",
        ]
        if system:
            cmd += ["--system-prompt", system]
        if json_schema:
            cmd += ["--json-schema", json_schema]
        if allowed_tools:
            cmd += ["--allowedTools", ",".join(allowed_tools)]

        # Strip ANTHROPIC_API_KEY so the claude CLI authenticates via OAuth
        # (Claude Code Max subscription) instead of the low-credit API key.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_override or self.timeout,
            env=env,
        )
        if result.returncode != 0:
            # The claude CLI sometimes writes error messages to stdout instead of stderr
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            error_text = stderr or stdout
            if any(p in error_text.lower() for p in _RATE_LIMIT_PATTERNS):
                raise ClaudeRateLimitError(
                    f"Claude Code usage limit reached: {error_text}"
                )
            raise RuntimeError(f"claude CLI exited {result.returncode}: {error_text}")
        return result.stdout.strip()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, prompt = self._format_messages(messages)
        content = self._call_cli(system, prompt)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    def with_structured_output(  # type: ignore[override]
        self,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> Any:
        """Return a runnable that injects JSON schema into the prompt.

        The ``--json-schema`` CLI flag does not enforce JSON output, so we
        instead append the schema and an explicit JSON-only instruction to
        the system prompt, then strip any markdown fences from the response.
        """
        from langchain_core.runnables import RunnableLambda

        schema_str = json.dumps(schema.model_json_schema(), indent=2)
        json_instruction = (
            "\n\nYou MUST respond with ONLY valid JSON that matches this schema "
            f"(no explanation, no markdown fences):\n{schema_str}"
        )
        model = self

        def invoke_structured(messages: list[BaseMessage]) -> BaseModel:
            system, prompt = model._format_messages(messages)
            augmented_system = system + json_instruction
            raw = model._call_cli(augmented_system, prompt)
            # Strip markdown code fences if present
            stripped = raw.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
                stripped = stripped.rsplit("```", 1)[0].strip()
            return schema.model_validate_json(stripped)

        return RunnableLambda(invoke_structured)
