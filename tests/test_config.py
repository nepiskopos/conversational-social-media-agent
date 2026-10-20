"""
Unit tests for agent/config.py.

Owns: provider selection in registry order and LLM_PROVIDER, each provider's
model variable, custom LiteLLM models, fallbacks, the fast and judge models
(FAST_PROVIDER, JUDGE_PROVIDER), connection_kwargs and background_kwargs, and
the errors and warnings a mismatched or ambiguous setting logs.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import agent.config as config
from agent.config import (
    LLM_TIMEOUT_S,
    PROVIDERS,
    background_kwargs,
    connection_kwargs,
    is_anthropic_model,
    model_matches_provider,
    resolve_fallbacks,
    resolve_model,
)

TIMEOUT: dict = {"timeout": LLM_TIMEOUT_S}


def _logged(caplog) -> str:
    """What config logged since the last call (warnings and errors), then forget it."""
    text = caplog.text
    caplog.clear()
    return text


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """Warnings print once per process; each test starts as if none had been printed."""
    from agent.config import _report_unusable_provider, _warn_ambiguous_model

    _report_unusable_provider.cache_clear()
    _warn_ambiguous_model.cache_clear()


def _env(**kwargs: str) -> dict[str, str]:
    """
    Build a full os.environ patch: blank every provider variable, then apply overrides.

    An empty string is falsy in os.getenv(), so blanked variables count as unset.
    This isolates tests from whatever the caller's shell has set.
    """
    keys = {"LLM_PROVIDER", "FAST_MODEL", "FAST_PROVIDER", "JUDGE_MODEL", "JUDGE_PROVIDER"}
    for provider in PROVIDERS:
        keys |= {provider.key_env, provider.model_env, *provider.requires}
        if provider.base_url_env:
            keys.add(provider.base_url_env)
    base: dict[str, str] = {k: "" for k in keys}
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# No keys set: the Anthropic default
# ---------------------------------------------------------------------------


def test_no_keys_returns_anthropic_default():
    with patch.dict("os.environ", _env(), clear=False):
        assert resolve_model() == "claude-haiku-5-5"


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

NAMED_PROVIDERS = [provider for provider in PROVIDERS if provider.id != "custom"]


def test_providers_are_chosen_in_registry_order():
    """
    With every key set, each provider wins once all those above it are unset.

    The last round is the bottom provider alone, so every provider is also
    shown to work on its own key(s) with its default model.
    """
    # The order the README documents; changing it changes which model users get.
    documented = ["anthropic", "openai", "xai", "google", "mistral", "groq", "deepseek"]
    documented += ["openrouter", "opencode", "qwen"]
    assert [provider.id for provider in NAMED_PROVIDERS] == documented
    for rank, provider in enumerate(NAMED_PROVIDERS):
        keys = {name: "x" for below in NAMED_PROVIDERS[rank:] for name in below.requires}
        with patch.dict("os.environ", _env(**keys)):
            assert resolve_model() == provider.default_model, provider.id


def test_each_providers_model_variable_replaces_its_default(caplog):
    for provider in NAMED_PROVIDERS:
        custom = f"{provider.default_model}-custom"
        env = _env(**{name: "x" for name in provider.requires}, **{provider.model_env: custom})
        with patch.dict("os.environ", env):
            assert resolve_model() == custom, provider.id
    assert _logged(caplog) == ""


# ---------------------------------------------------------------------------
# A model that doesn't match its provider
# ---------------------------------------------------------------------------


def test_mismatch_prints_an_error_and_still_uses_the_configured_model(caplog):
    with patch.dict(
        "os.environ",
        _env(ANTHROPIC_API_KEY="sk-ant-abc", ANTHROPIC_MODEL="gpt-4o-mini"),
        clear=False,
    ):
        model = resolve_model()
    assert model == "gpt-4o-mini"
    err = _logged(caplog)
    assert "ERROR" in err
    assert "ANTHROPIC_MODEL" in err
    assert "gpt-4o-mini" in err


# ---------------------------------------------------------------------------
# is_anthropic_model / model_matches_provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-haiku-5-5", True),
        ("anthropic/claude-haiku-5-5", True),
        ("gpt-4o-mini", False),
        # Bedrock models must NOT match: they use AWS-side caching.
        ("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", False),
        ("us.anthropic.claude-3-5-sonnet-20241022-v2:0", False),
    ],
)
def test_is_anthropic_model(model, expected):
    assert is_anthropic_model(model) is expected


@pytest.mark.parametrize(
    ("provider_id", "its_model", "another_model"),
    [
        ("openai", "gpt-4o-mini", "claude-haiku-5-5"),
        ("google", "gemini/gemini-2.0-flash", "gpt-4o-mini"),
    ],
)
def test_model_matches_provider(provider_id, its_model, another_model):
    provider = next(p for p in PROVIDERS if p.id == provider_id)
    assert model_matches_provider(its_model, provider) is True
    assert model_matches_provider(another_model, provider) is False


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


def test_resolve_fallbacks_lists_other_configured_providers_in_priority_order():
    env = _env(
        ANTHROPIC_API_KEY="a", OPENAI_API_KEY="o", GEMINI_API_KEY="g", OPENAI_MODEL="gpt-4.1"
    )
    with patch.dict("os.environ", env):
        assert resolve_fallbacks("claude-haiku-5-5") == [
            "gpt-4.1",
            "gemini/gemini-2.0-flash",
        ]


def test_resolve_fallbacks_empty_with_a_single_key():
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="a")):
        assert resolve_fallbacks("claude-haiku-5-5") == []


# ---------------------------------------------------------------------------
# More providers: OpenAI-compatible endpoints, LLM_PROVIDER, custom LiteLLM models
# ---------------------------------------------------------------------------


def test_opencode_zen_uses_its_endpoint_and_key():
    with patch.dict("os.environ", _env(OPENCODE_API_KEY="zen-key")):
        model = resolve_model()
        assert model == "openai/glm-5.3-flash"
        assert connection_kwargs(model) == {
            **TIMEOUT,
            "api_base": "https://opencode.ai/zen/v1",
            "api_key": "zen-key",
        }
    env = _env(OPENCODE_API_KEY="zen-key", OPENCODE_BASE_URL="https://zen.internal/v1")
    with patch.dict("os.environ", env):
        assert connection_kwargs("openai/glm-5.3-flash")["api_base"] == "https://zen.internal/v1"


def test_qwen_needs_both_url_and_key():
    with patch.dict("os.environ", _env(QWEN_API_KEY="q")):
        assert resolve_model() == "claude-haiku-5-5"  # URL missing → not configured
    with patch.dict("os.environ", _env(QWEN_API_KEY="q", QWEN_BASE_URL="http://gpu-box:8000/v1")):
        model = resolve_model()
        assert model == "hosted_vllm/qwen36-35b-a3b"
        assert connection_kwargs(model) == {
            **TIMEOUT,
            "api_base": "http://gpu-box:8000/v1",
            "api_key": "q",
        }


def test_litellm_prefix_added_to_a_bare_model_name(caplog):
    with patch.dict(
        "os.environ", _env(QWEN_API_KEY="q", QWEN_BASE_URL="http://x/v1", QWEN_MODEL="qwen3-8b")
    ):
        assert resolve_model() == "hosted_vllm/qwen3-8b"
    with patch.dict("os.environ", _env(GROQ_API_KEY="g", GROQ_MODEL="groq/llama-3.1-8b-instant")):
        assert resolve_model() == "groq/llama-3.1-8b-instant"
    # LiteLLM can't route these bare names on its own; every call failed before.
    with patch.dict("os.environ", _env(XAI_API_KEY="x", XAI_MODEL="grok-3-mini")):
        assert resolve_model() == "xai/grok-3-mini"
    with patch.dict("os.environ", _env(GEMINI_API_KEY="g", GEMINI_MODEL="gemini-2.0-flash")):
        assert resolve_model() == "gemini/gemini-2.0-flash"
    # OpenAI's bare names route natively; newer families must not be flagged as mismatches.
    with patch.dict("os.environ", _env(OPENAI_API_KEY="o", OPENAI_MODEL="o4-mini")):
        assert resolve_model() == "o4-mini"
    assert _logged(caplog) == ""


def test_llm_provider_moves_a_configured_provider_first():
    env = _env(ANTHROPIC_API_KEY="a", OPENCODE_API_KEY="z", LLM_PROVIDER="opencode")
    with patch.dict("os.environ", env):
        assert resolve_model() == "openai/glm-5.3-flash"
        assert resolve_fallbacks("openai/glm-5.3-flash") == ["claude-haiku-5-5"]


def test_llm_provider_unknown_or_unconfigured_prints_error_and_keeps_default_order(caplog):
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="a", LLM_PROVIDER="qwen")):
        assert resolve_model() == "claude-haiku-5-5"
    err = _logged(caplog)
    assert "LLM_PROVIDER" in err and "qwen" in err


def test_custom_model_accepts_any_litellm_string_and_takes_priority(caplog):
    env = _env(
        ANTHROPIC_API_KEY="a",
        LLM_MODEL="ollama_chat/llama3.1",
        LLM_API_BASE="http://localhost:11434",
    )
    with patch.dict("os.environ", env):
        model = resolve_model()
        assert model == "ollama_chat/llama3.1"
        assert connection_kwargs(model) == {**TIMEOUT, "api_base": "http://localhost:11434"}
        assert resolve_fallbacks(model) == ["claude-haiku-5-5"]
    assert _logged(caplog) == ""


def test_connection_kwargs_matches_a_fast_model_by_prefix():
    env = _env(QWEN_API_KEY="q", QWEN_BASE_URL="http://x/v1")
    with patch.dict("os.environ", env):
        assert connection_kwargs("hosted_vllm/qwen3-8b") == {
            **TIMEOUT,
            "api_base": "http://x/v1",
            "api_key": "q",
        }


def test_connection_kwargs_empty_for_an_unconfigured_model():
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="a")):
        assert connection_kwargs("gpt-4o-mini") == TIMEOUT
        assert connection_kwargs("claude-haiku-5-5") == {**TIMEOUT, "api_key": "a"}


def test_background_kwargs_turn_down_reasoning_for_background_work():
    env = _env(
        QWEN_API_KEY="q", QWEN_BASE_URL="http://x/v1", OPENCODE_API_KEY="z", ANTHROPIC_API_KEY="a"
    )
    with patch.dict("os.environ", env):
        qwen = background_kwargs("hosted_vllm/qwen36-35b-a3b")
        assert qwen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
        assert qwen["api_base"] == "http://x/v1"
        assert background_kwargs("openai/glm-5.3-flash")["reasoning_effort"] == "low"
        assert background_kwargs("claude-haiku-5-5") == {
            **TIMEOUT,
            "api_key": "a",
        }  # nothing extra
        assert "extra_body" not in connection_kwargs(
            "hosted_vllm/qwen36-35b-a3b"
        )  # user-facing calls keep reasoning


def test_a_model_two_configured_providers_could_serve_warns_once(caplog):
    """openai/ is both OpenAI's prefix and OpenCode Zen's: the choice must not be silent."""
    with patch.dict("os.environ", _env(OPENAI_API_KEY="o", OPENCODE_API_KEY="z")):
        for _ in range(3):
            assert connection_kwargs("openai/deepseek-v4-pro")["api_key"] == "o"
        assert connection_kwargs("openai/glm-5.3-flash")["api_key"] == "z"  # exact: no doubt
    err = _logged(caplog)
    assert err.count("WARNING") == 1
    assert "'openai/deepseek-v4-pro' could belong to openai or opencode" in err
    assert "JUDGE_PROVIDER" in err  # the warning says how to choose


# The fast and judge models: each has a model variable and a provider variable.
ROLES = [
    ("FAST_MODEL", "FAST_PROVIDER", "resolve_fast_model"),
    ("JUDGE_MODEL", "JUDGE_PROVIDER", "resolve_judge_model"),
]


@pytest.mark.parametrize(("model_var", "provider_var", "resolver"), ROLES)
def test_a_named_provider_serves_its_roles_model_without_a_warning(
    model_var, provider_var, resolver, caplog
):
    """openai/ is OpenAI's and OpenCode Zen's; naming Zen sends the call there."""
    env = _env(
        OPENAI_API_KEY="o",
        OPENCODE_API_KEY="z",
        **{model_var: "deepseek-v4-pro", provider_var: "OpenCode"},
    )
    with patch.dict("os.environ", env):
        model = getattr(config, resolver)()
        assert model == "openai/deepseek-v4-pro"  # Zen's LiteLLM prefix added
        assert connection_kwargs(model) == {
            **TIMEOUT,
            "api_base": "https://opencode.ai/zen/v1",
            "api_key": "z",
        }
        assert background_kwargs(model)["reasoning_effort"] == "low"
        assert resolve_model() == "gpt-4o-mini"  # the main model is untouched
        assert connection_kwargs("gpt-4o-mini")["api_key"] == "o"
    assert _logged(caplog) == ""


@pytest.mark.parametrize(("model_var", "provider_var", "resolver"), ROLES)
def test_a_roles_model_comes_from_its_provider_its_own_variable_or_the_main_model(
    model_var, provider_var, resolver
):
    env = _env(ANTHROPIC_API_KEY="a", GROQ_API_KEY="g", **{provider_var: "groq"})
    with patch.dict("os.environ", env):
        assert getattr(config, resolver)() == "groq/llama-3.3-70b-versatile"
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="a", **{model_var: "gpt-4o-mini"})):
        assert getattr(config, resolver)() == "gpt-4o-mini"  # no provider named: used verbatim
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="a")):
        assert getattr(config, resolver)() == "claude-haiku-5-5"  # nothing set: the main model


@pytest.mark.parametrize(("model_var", "provider_var", "resolver"), ROLES)
def test_an_unusable_provider_name_is_reported_once_and_ignored(
    model_var, provider_var, resolver, caplog
):
    env = _env(
        OPENAI_API_KEY="o", **{model_var: "openai/deepseek-v4-pro", provider_var: "opencode"}
    )
    with patch.dict("os.environ", env):
        for _ in range(3):
            model = getattr(config, resolver)()
            assert model == "openai/deepseek-v4-pro"
            assert connection_kwargs(model)["api_key"] == "o"
    err = _logged(caplog)
    assert err.count("ERROR") == 1
    assert f"{provider_var}='opencode' is not a configured provider" in err


def test_a_providers_own_model_stays_with_it_whatever_a_role_names():
    """The main model and its fallbacks are never rerouted by FAST_PROVIDER."""
    env = _env(
        OPENAI_API_KEY="o",
        OPENCODE_API_KEY="z",
        FAST_MODEL="openai/glm-5.3-flash",
        FAST_PROVIDER="openai",
    )
    with patch.dict("os.environ", env):
        assert connection_kwargs("openai/glm-5.3-flash")["api_key"] == "z"


def test_load_settings_applies_the_time_zone_in_the_env_file(tmp_path, monkeypatch):
    """Regression: Python reads TZ at start, so a TZ from .env changed nothing outside Docker."""
    import time
    from datetime import datetime

    from agent.config import load_settings

    env_file = tmp_path / ".env"
    env_file.write_text("TZ=America/New_York\n")
    monkeypatch.delenv("TZ")
    try:
        load_settings(env_file)
        assert datetime(2026, 1, 15, 12).astimezone().utcoffset().total_seconds() == -5 * 3600
    finally:
        monkeypatch.setenv("TZ", "UTC")
        time.tzset()


def test_a_blank_tz_in_the_env_file_leaves_the_servers_zone(tmp_path, monkeypatch):
    """Regression: .env.example's blank "TZ=" switched the server to UTC instead."""
    import os
    import time

    from agent.config import load_settings

    env_file = tmp_path / ".env"
    env_file.write_text("TZ=\n")
    monkeypatch.delenv("TZ")
    try:
        load_settings(env_file)
        assert "TZ" not in os.environ
    finally:
        monkeypatch.setenv("TZ", "UTC")
        time.tzset()
