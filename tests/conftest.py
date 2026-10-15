from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.interpreter import OpenAICompatInterpreter
from app.main import create_app


def _say(client: TestClient) -> Callable[..., dict[str, Any]]:
    def say(text: str, session_id: str | None = None, mode: str = "auto", clinician: str = "dr_demo") -> dict[str, Any]:
        r = client.post("/api/assistant/message", json={"text": text, "session_id": session_id, "mode": mode},
                        headers={"X-Clinician-Id": clinician})
        assert r.status_code == 200, r.text
        return r.json()
    return say


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings()))


@pytest.fixture
def say(client: TestClient) -> Callable[..., dict[str, Any]]:
    return _say(client)


class FakeLLMClient:
    """Stands in for an OpenAI-compatible client (Groq): returns a scripted tool call, or raises."""

    def __init__(self, tool_name: str | None = None, tool_input: dict[str, Any] | None = None,
                 error: Exception | None = None):
        self.calls: list[dict[str, Any]] = []
        self.tool_name, self.tool_input, self.error = tool_name, tool_input or {}, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        call = SimpleNamespace(function=SimpleNamespace(name=self.tool_name, arguments=json.dumps(self.tool_input)))
        message = SimpleNamespace(tool_calls=[call], content=None)
        usage = SimpleNamespace(prompt_tokens=830, completion_tokens=96, prompt_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture
def llm_app() -> Callable[..., tuple[TestClient, FakeLLMClient]]:
    def make(**fake_kwargs: Any) -> tuple[TestClient, FakeLLMClient]:
        fake = FakeLLMClient(**fake_kwargs)
        interp = OpenAICompatInterpreter(None, "openai/gpt-oss-120b", "https://api.groq.com/openai/v1",
                                         client=fake, provider="groq")
        return TestClient(create_app(Settings(), interpreter=interp)), fake
    return make


@pytest.fixture
def llm_say() -> Callable[[TestClient], Callable[..., dict[str, Any]]]:
    return _say
