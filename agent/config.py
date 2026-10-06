"""
Provider selection and model resolution for Mackie.

Owns: the provider registry (PROVIDERS) and the mapping from environment
variables (API keys, model names, endpoint URLs) to the LiteLLM model string
and connection settings (api_base, api_key) for each call. Does not own any
LLM call logic — it only reads os.environ.

A provider counts as configured when every variable in its `requires` is set.
LLM_PROVIDER picks which configured provider goes first; otherwise the
registry order below decides. Every other configured provider is a fallback.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class _Provider:
    """
    Static descriptor for one LLM provider.

    model_prefixes:  prefixes a valid model string for this provider starts with;
                     empty means any string is accepted (no mismatch check).
    litellm_prefix:  added in front of the configured model when missing, so a
                     user can write QWEN_MODEL=qwen36-35b-a3b instead of
                     hosted_vllm/qwen36-35b-a3b.
    base_url_env:    variable holding the endpoint URL (OpenAI-compatible servers).
    default_base_url: endpoint used when base_url_env is unset.
    requires:        variables that must all be set for the provider to be used.
    background_kwargs: extra LiteLLM arguments for background calls (summaries,
                     org brief, calendar, eval judge) that keep a reasoning model
                     from thinking at length about a simple writing job.
    """

    id: str
    key_env: str
    model_env: str
    model_prefixes: tuple[str, ...]
    default_model: str
    requires: tuple[str, ...]
    litellm_prefix: str = ""
    base_url_env: str = ""
    default_base_url: str = ""
    background_kwargs: dict[str, Any] = field(default_factory=dict, compare=False)


# Registry order is the default priority. "custom" is first because setting
# LLM_MODEL is an explicit choice. This list is the single source of truth for
# provider metadata; import from here rather than repeating it in code.
PROVIDERS: list[_Provider] = [
    _Provider(  # any model LiteLLM supports (Azure, Bedrock, Vertex, Ollama, Together, ...)
        id="custom",
        key_env="LLM_API_KEY",
        model_env="LLM_MODEL",
        model_prefixes=(),
        default_model="",
        requires=("LLM_MODEL",),
        base_url_env="LLM_API_BASE",
    ),
    _Provider(
        id="anthropic",
        key_env="ANTHROPIC_API_KEY",
        model_env="ANTHROPIC_MODEL",
        model_prefixes=("claude", "anthropic/"),
        default_model="claude-haiku-4-5-20251001",
        requires=("ANTHROPIC_API_KEY",),
    ),
    _Provider(
        id="openai",
        key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        model_prefixes=("gpt-", "o1-", "o3-", "openai/"),
        default_model="gpt-4o-mini",
        requires=("OPENAI_API_KEY",),
    ),
    _Provider(
        id="xai",
        key_env="XAI_API_KEY",
        model_env="XAI_MODEL",
        model_prefixes=("grok-", "xai/"),
        default_model="xai/grok-3-mini",
        requires=("XAI_API_KEY",),
    ),
    _Provider(
        id="google",
        key_env="GEMINI_API_KEY",
        model_env="GEMINI_MODEL",
        model_prefixes=("gemini/",),
        default_model="gemini/gemini-2.0-flash",
        requires=("GEMINI_API_KEY",),
    ),
    _Provider(
        id="mistral",
        key_env="MISTRAL_API_KEY",
        model_env="MISTRAL_MODEL",
        model_prefixes=("mistral/",),
        default_model="mistral/mistral-small-latest",
        requires=("MISTRAL_API_KEY",),
        litellm_prefix="mistral/",
    ),
    _Provider(
        id="groq",
        key_env="GROQ_API_KEY",
        model_env="GROQ_MODEL",
        model_prefixes=("groq/",),
        default_model="groq/llama-3.3-70b-versatile",
        requires=("GROQ_API_KEY",),
        litellm_prefix="groq/",
    ),
    _Provider(
        id="deepseek",
        key_env="DEEPSEEK_API_KEY",
        model_env="DEEPSEEK_MODEL",
        model_prefixes=("deepseek/",),
        default_model="deepseek/deepseek-chat",
        requires=("DEEPSEEK_API_KEY",),
        litellm_prefix="deepseek/",
    ),
    _Provider(
        id="openrouter",
        key_env="OPENROUTER_API_KEY",
        model_env="OPENROUTER_MODEL",
        model_prefixes=("openrouter/",),
        default_model="openrouter/openai/gpt-4o-mini",
        requires=("OPENROUTER_API_KEY",),
        litellm_prefix="openrouter/",
    ),
    _Provider(  # OpenCode Zen — OpenAI-compatible; free models may be used to train the provider's models
        id="opencode",
        key_env="OPENCODE_API_KEY",
        model_env="OPENCODE_MODEL",
        model_prefixes=("openai/",),
        default_model="openai/space-bunny-free",
        requires=("OPENCODE_API_KEY",),
        litellm_prefix="openai/",
        base_url_env="OPENCODE_BASE_URL",
        default_base_url="https://opencode.ai/zen/v1",
        # Measured on the org brief: 2,112 reasoning tokens / 20 s by default, 59 / 5 s at "low".
        background_kwargs={"reasoning_effort": "low"},
    ),
    _Provider(  # self-hosted Qwen behind vLLM; the URL stays in .env, never in the repo
        id="qwen",
        key_env="QWEN_API_KEY",
        model_env="QWEN_MODEL",
        model_prefixes=("hosted_vllm/",),
        default_model="hosted_vllm/qwen36-35b-a3b",
        requires=("QWEN_BASE_URL", "QWEN_API_KEY"),
        litellm_prefix="hosted_vllm/",
        base_url_env="QWEN_BASE_URL",
        # Qwen 3 thinks until any token cap runs out (no brief even at 4,096 tokens);
        # with thinking off it wrote the brief in 750 tokens. reasoning_effort is ignored.
        background_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    ),
]


def model_matches_provider(model: str, provider: _Provider) -> bool:
    """Return True if the model string is compatible with the given provider."""
    return any(model.startswith(prefix) for prefix in provider.model_prefixes)


def is_anthropic_model(model: str) -> bool:
    """
    Return True if model targets the native Anthropic API.

    Used by graph.py to decide whether to send the system prompt as content
    blocks with a cache_control breakpoint. Bedrock ("bedrock/anthropic.claude-*",
    "us.anthropic.claude-*"), Vertex AI and OpenRouter routes are intentionally
    excluded and receive a plain string system prompt.
    """
    anthropic = next(p for p in PROVIDERS if p.id == "anthropic")
    return model_matches_provider(model, anthropic)


def _is_configured(provider: _Provider) -> bool:
    """True when every variable the provider requires is set (non-empty)."""
    return all(os.getenv(name) for name in provider.requires)


def _provider_model(provider: _Provider) -> str:
    """The provider's configured (or default) model, with its LiteLLM prefix added."""
    model = os.getenv(provider.model_env) or provider.default_model
    if provider.litellm_prefix and not model.startswith(provider.litellm_prefix):
        model = provider.litellm_prefix + model
    return model


def _configured_providers() -> list[_Provider]:
    """
    Configured providers, highest priority first.

    LLM_PROVIDER (a provider id) moves that provider to the front; an unknown
    or unconfigured id prints an error and the registry order is used instead.
    """
    configured = [p for p in PROVIDERS if _is_configured(p)]
    preferred = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not preferred:
        return configured
    chosen = next((p for p in configured if p.id == preferred), None)
    if chosen is None:
        known = ", ".join(p.id for p in PROVIDERS)
        print(
            f"ERROR: LLM_PROVIDER={preferred!r} is not a configured provider "
            f"(known: {known}; its key variables must be set). Using the default order.",
            file=sys.stderr,
        )
        return configured
    return [chosen] + [p for p in configured if p is not chosen]


def resolve_model() -> str:
    """
    Resolve the LLM model to use from environment variables.

    Returns the model of the highest-priority configured provider (see
    _configured_providers). Prints an error to stderr if the configured model
    doesn't match its provider (e.g. ANTHROPIC_API_KEY set but
    ANTHROPIC_MODEL=gpt-4o-mini).

    When no provider is configured (e.g. during tests), falls back to the
    Anthropic default.

    Returns:
        A LiteLLM-compatible model string ready to pass to litellm.acompletion()
        together with connection_kwargs(model).
    """
    available = _configured_providers()
    if not available:
        return next(p for p in PROVIDERS if p.id == "anthropic").default_model

    selected = available[0]
    model = _provider_model(selected)

    if selected.model_prefixes and not model_matches_provider(model, selected):
        print(
            f"ERROR: {selected.model_env}={model!r} does not match provider "
            f"'{selected.id}' (expected prefix in {selected.model_prefixes}). "
            f"Check your .env configuration.",
            file=sys.stderr,
        )

    return model


def resolve_fallbacks(primary: str) -> list[str]:
    """
    Return backup models to try, in priority order, if the primary model fails.

    Every other configured provider contributes its configured (or default)
    model, so adding a second key is all it takes to survive one provider's
    outage. Returns an empty list when only one provider is configured.

    Args:
        primary: The model already in use; it is never repeated as a fallback.

    Returns:
        LiteLLM model strings, highest-priority provider first.
    """
    models = [_provider_model(p) for p in _configured_providers()]
    return [m for m in dict.fromkeys(models) if m != primary]


def _provider_for(model: str) -> _Provider | None:
    """The configured provider whose model is exactly `model`, else the first whose prefixes match."""
    configured = _configured_providers()
    return next((p for p in configured if _provider_model(p) == model), None) or next(
        (p for p in configured if p.model_prefixes and model_matches_provider(model, p)), None
    )


def background_kwargs(model: str) -> dict[str, Any]:
    """
    Return connection_kwargs(model) plus the provider's background_kwargs.

    For calls that do a short, well-specified job — the conversation summary,
    the org brief, calendar plans, the eval judge. Reasoning models otherwise
    spend the whole token budget thinking and return no text. User-facing
    replies use connection_kwargs() alone and keep the model's full reasoning.
    """
    provider = _provider_for(model)
    extra = dict(provider.background_kwargs) if provider else {}
    return {**connection_kwargs(model), **extra}


def connection_kwargs(model: str) -> dict[str, str]:
    """
    Return the api_base / api_key to pass to litellm.acompletion() with model.

    The provider is the configured one whose model is exactly `model`, else the
    first configured one whose prefixes match it. Native providers need nothing
    here when their key uses LiteLLM's own variable name, but passing it is
    harmless; OpenAI-compatible endpoints (OpenCode Zen, vLLM, LLM_API_BASE)
    need both values, since LiteLLM cannot guess them. Unknown models get {}
    and LiteLLM reads its usual environment variables.

    Args:
        model: A LiteLLM model string, e.g. from resolve_model() or FAST_MODEL.

    Returns:
        Dict with "api_base" and/or "api_key", only for values that are set.
    """
    provider = _provider_for(model)
    if provider is None:
        return {}
    settings = {
        "api_base": (os.getenv(provider.base_url_env) if provider.base_url_env else "")
        or provider.default_base_url,
        "api_key": os.getenv(provider.key_env, ""),
    }
    return {name: value for name, value in settings.items() if value}


def resolve_fast_model() -> str:
    """
    Return the model for background work: summaries, org briefs, calendar plans.

    FAST_MODEL lets a deployment route that work to a cheaper model (cost
    routing) while user-facing drafts keep the main model. Unset, it is the
    same as resolve_model(). The provider for FAST_MODEL must be configured.
    """
    return os.getenv("FAST_MODEL") or resolve_model()
