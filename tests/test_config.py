"""Unit tests for agent/config.py — provider selection and model resolution."""

from __future__ import annotations

from unittest.mock import patch

from agent.config import (
    PROVIDERS,
    connection_kwargs,
    is_anthropic_model,
    model_matches_provider,
    resolve_fallbacks,
    resolve_model,
)


def _env(**kwargs: str) -> dict[str, str]:
    """Build a full os.environ patch: unset all relevant vars, then apply overrides.

    Empty string is falsy in os.getenv(), so unset vars are treated as absent
    by resolve_model(). This isolates tests from whatever the caller's shell has set.
    """
    keys = {"LLM_PROVIDER", "FAST_MODEL"}
    for provider in PROVIDERS:
        keys |= {provider.key_env, provider.model_env, *provider.requires}
        if provider.base_url_env:
            keys.add(provider.base_url_env)
    base: dict[str, str] = {k: "" for k in keys}
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# No keys set — offline / test fallback
# ---------------------------------------------------------------------------


def test_no_keys_returns_anthropic_default():
    with patch.dict("os.environ", _env(), clear=False):
        assert resolve_model() == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Single provider selected
# ---------------------------------------------------------------------------


def test_anthropic_key_returns_default_anthropic_model():
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="sk-ant-abc"), clear=False):
        assert resolve_model() == "claude-haiku-4-5-20251001"


def test_anthropic_key_with_custom_model():
    with patch.dict(
        "os.environ",
        _env(ANTHROPIC_API_KEY="sk-ant-abc", ANTHROPIC_MODEL="claude-opus-4-8"),
        clear=False,
    ):
        assert resolve_model() == "claude-opus-4-8"


def test_openai_key_returns_default_openai_model():
    with patch.dict("os.environ", _env(OPENAI_API_KEY="sk-abc"), clear=False):
        assert resolve_model() == "gpt-4o-mini"


def test_openai_key_with_custom_model():
    with patch.dict(
        "os.environ", _env(OPENAI_API_KEY="sk-abc", OPENAI_MODEL="gpt-4o"), clear=False
    ):
        assert resolve_model() == "gpt-4o"


def test_xai_key_returns_default_xai_model():
    with patch.dict("os.environ", _env(XAI_API_KEY="xai-abc"), clear=False):
        assert resolve_model() == "xai/grok-3-mini"


def test_gemini_key_returns_default_gemini_model():
    with patch.dict("os.environ", _env(GEMINI_API_KEY="AIzaabc"), clear=False):
        assert resolve_model() == "gemini/gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Priority: multiple keys set → Anthropic wins
# ---------------------------------------------------------------------------


def test_anthropic_beats_openai_when_both_keys_set():
    with patch.dict(
        "os.environ", _env(ANTHROPIC_API_KEY="sk-ant-abc", OPENAI_API_KEY="sk-abc"), clear=False
    ):
        assert resolve_model().startswith("claude")


def test_anthropic_beats_all_providers():
    with patch.dict(
        "os.environ",
        _env(
            ANTHROPIC_API_KEY="sk-ant-abc",
            OPENAI_API_KEY="sk-abc",
            XAI_API_KEY="xai-abc",
            GEMINI_API_KEY="AIzaabc",
        ),
        clear=False,
    ):
        assert resolve_model().startswith("claude")


def test_openai_beats_xai_and_google():
    with patch.dict(
        "os.environ",
        _env(OPENAI_API_KEY="sk-abc", XAI_API_KEY="xai-abc", GEMINI_API_KEY="AIzaabc"),
        clear=False,
    ):
        assert resolve_model().startswith("gpt-")


def test_xai_beats_google():
    with patch.dict(
        "os.environ", _env(XAI_API_KEY="xai-abc", GEMINI_API_KEY="AIzaabc"), clear=False
    ):
        assert resolve_model().startswith("xai/")


# ---------------------------------------------------------------------------
# Mismatch error
# ---------------------------------------------------------------------------


def test_mismatch_prints_error(capsys):
    with patch.dict(
        "os.environ",
        _env(ANTHROPIC_API_KEY="sk-ant-abc", ANTHROPIC_MODEL="gpt-4o-mini"),
        clear=False,
    ):
        resolve_model()
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "ANTHROPIC_MODEL" in err
    assert "gpt-4o-mini" in err


def test_mismatch_still_returns_the_configured_model(capsys):
    with patch.dict(
        "os.environ",
        _env(ANTHROPIC_API_KEY="sk-ant-abc", ANTHROPIC_MODEL="gpt-4o-mini"),
        clear=False,
    ):
        model = resolve_model()
    assert model == "gpt-4o-mini"


def test_correct_model_no_error(capsys):
    with patch.dict(
        "os.environ",
        _env(ANTHROPIC_API_KEY="sk-ant-abc", ANTHROPIC_MODEL="claude-opus-4-8"),
        clear=False,
    ):
        resolve_model()
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# is_anthropic_model / model_matches_provider
# ---------------------------------------------------------------------------


def test_is_anthropic_model_native_claude():
    assert is_anthropic_model("claude-haiku-4-5-20251001") is True


def test_is_anthropic_model_with_anthropic_prefix():
    assert is_anthropic_model("anthropic/claude-haiku-4-5-20251001") is True


def test_is_anthropic_model_rejects_openai():
    assert is_anthropic_model("gpt-4o-mini") is False


def test_is_anthropic_model_rejects_bedrock():
    # Bedrock cross-region and bedrock/ prefix must NOT match — they use AWS-side caching.
    assert is_anthropic_model("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0") is False
    assert is_anthropic_model("us.anthropic.claude-3-5-sonnet-20241022-v2:0") is False


def test_model_matches_provider_openai():
    openai = next(p for p in PROVIDERS if p.id == "openai")
    assert model_matches_provider("gpt-4o-mini", openai) is True
    assert model_matches_provider("claude-haiku-4-5-20251001", openai) is False


def test_model_matches_provider_google():
    google = next(p for p in PROVIDERS if p.id == "google")
    assert model_matches_provider("gemini/gemini-2.0-flash", google) is True
    assert model_matches_provider("gpt-4o-mini", google) is False


# ---------------------------------------------------------------------------
# Fallbacks and the fast model
# ---------------------------------------------------------------------------


def test_resolve_fallbacks_lists_other_configured_providers_in_priority_order():
    from agent.config import resolve_fallbacks

    env = _env(
        ANTHROPIC_API_KEY="a", OPENAI_API_KEY="o", GEMINI_API_KEY="g", OPENAI_MODEL="gpt-4.1"
    )
    with patch.dict("os.environ", env):
        assert resolve_fallbacks("claude-haiku-4-5-20251001") == [
            "gpt-4.1",
            "gemini/gemini-2.0-flash",
        ]


def test_resolve_fallbacks_empty_with_a_single_key():
    from agent.config import resolve_fallbacks

    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="a")):
        assert resolve_fallbacks("claude-haiku-4-5-20251001") == []


def test_resolve_fast_model_uses_env_override_or_main_model():
    from agent.config import resolve_fast_model

    with patch.dict("os.environ", {**_env(ANTHROPIC_API_KEY="a"), "FAST_MODEL": ""}):
        assert resolve_fast_model() == "claude-haiku-4-5-20251001"
    with patch.dict("os.environ", {**_env(ANTHROPIC_API_KEY="a"), "FAST_MODEL": "gpt-4o-mini"}):
        assert resolve_fast_model() == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# More providers: OpenAI-compatible endpoints, LLM_PROVIDER, custom LiteLLM models
# ---------------------------------------------------------------------------


def test_opencode_zen_uses_its_endpoint_and_key():
    with patch.dict("os.environ", _env(OPENCODE_API_KEY="zen-key")):
        model = resolve_model()
        assert model == "openai/space-bunny-free"
        assert connection_kwargs(model) == {
            "api_base": "https://opencode.ai/zen/v1",
            "api_key": "zen-key",
        }


def test_qwen_needs_both_url_and_key():
    with patch.dict("os.environ", _env(QWEN_API_KEY="q")):
        assert resolve_model() == "claude-haiku-4-5-20251001"  # URL missing → not configured
    with patch.dict("os.environ", _env(QWEN_API_KEY="q", QWEN_BASE_URL="http://gpu-box:8000/v1")):
        model = resolve_model()
        assert model == "hosted_vllm/qwen36-35b-a3b"
        assert connection_kwargs(model) == {"api_base": "http://gpu-box:8000/v1", "api_key": "q"}


def test_litellm_prefix_added_to_a_bare_model_name(capsys):
    with patch.dict(
        "os.environ", _env(QWEN_API_KEY="q", QWEN_BASE_URL="http://x/v1", QWEN_MODEL="qwen3-8b")
    ):
        assert resolve_model() == "hosted_vllm/qwen3-8b"
    with patch.dict("os.environ", _env(GROQ_API_KEY="g", GROQ_MODEL="groq/llama-3.1-8b-instant")):
        assert resolve_model() == "groq/llama-3.1-8b-instant"
    assert capsys.readouterr().err == ""


def test_each_named_provider_alone_resolves_to_its_default():
    for provider in PROVIDERS:
        if provider.id == "custom":
            continue
        env = _env(**{name: "x" for name in provider.requires})
        with patch.dict("os.environ", env):
            model = resolve_model()
        assert model == provider.default_model, provider.id


def test_llm_provider_moves_a_configured_provider_first():
    env = _env(ANTHROPIC_API_KEY="a", OPENCODE_API_KEY="z", LLM_PROVIDER="opencode")
    with patch.dict("os.environ", env):
        assert resolve_model() == "openai/space-bunny-free"
        assert resolve_fallbacks("openai/space-bunny-free") == ["claude-haiku-4-5-20251001"]


def test_llm_provider_unknown_or_unconfigured_prints_error_and_keeps_default_order(capsys):
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="a", LLM_PROVIDER="qwen")):
        assert resolve_model() == "claude-haiku-4-5-20251001"
    err = capsys.readouterr().err
    assert "LLM_PROVIDER" in err and "qwen" in err


def test_custom_model_accepts_any_litellm_string_and_takes_priority(capsys):
    env = _env(
        ANTHROPIC_API_KEY="a",
        LLM_MODEL="ollama_chat/llama3.1",
        LLM_API_BASE="http://localhost:11434",
    )
    with patch.dict("os.environ", env):
        model = resolve_model()
        assert model == "ollama_chat/llama3.1"
        assert connection_kwargs(model) == {"api_base": "http://localhost:11434"}
        assert resolve_fallbacks(model) == ["claude-haiku-4-5-20251001"]
    assert capsys.readouterr().err == ""


def test_connection_kwargs_matches_a_fast_model_by_prefix():
    env = _env(QWEN_API_KEY="q", QWEN_BASE_URL="http://x/v1")
    with patch.dict("os.environ", env):
        assert connection_kwargs("hosted_vllm/qwen3-8b") == {
            "api_base": "http://x/v1",
            "api_key": "q",
        }


def test_connection_kwargs_empty_for_an_unconfigured_model():
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="a")):
        assert connection_kwargs("gpt-4o-mini") == {}
        assert connection_kwargs("claude-haiku-4-5-20251001") == {"api_key": "a"}


def test_background_kwargs_turn_down_reasoning_for_background_work():
    from agent.config import background_kwargs

    env = _env(
        QWEN_API_KEY="q", QWEN_BASE_URL="http://x/v1", OPENCODE_API_KEY="z", ANTHROPIC_API_KEY="a"
    )
    with patch.dict("os.environ", env):
        qwen = background_kwargs("hosted_vllm/qwen36-35b-a3b")
        assert qwen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
        assert qwen["api_base"] == "http://x/v1"
        assert background_kwargs("openai/space-bunny-free")["reasoning_effort"] == "low"
        assert background_kwargs("claude-haiku-4-5-20251001") == {"api_key": "a"}  # nothing extra
        assert "extra_body" not in connection_kwargs(
            "hosted_vllm/qwen36-35b-a3b"
        )  # user-facing calls keep reasoning
