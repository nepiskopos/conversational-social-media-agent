"""
Provider selection and model resolution for Mackie.

Owns: the provider registry (PROVIDERS) and the mapping from environment
variables (API keys, model names, endpoint URLs) to the LiteLLM model string
and connection settings (api_base, api_key) for each call. Does not own any
LLM call logic — it only reads os.environ.

A provider counts as configured when every variable in its `requires` is set.
LLM_PROVIDER picks which configured provider goes first; otherwise the
registry order below decides. Every other configured provider is a fallback.
FAST_PROVIDER and JUDGE_PROVIDER name the provider for FAST_MODEL and
JUDGE_MODEL, for a model string more than one provider accepts.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Each background model's variable, and the variable naming its provider.
_ROLE_VARIABLES: tuple[tuple[str, str], ...] = (
    ("FAST_MODEL", "FAST_PROVIDER"),
    ("JUDGE_MODEL", "JUDGE_PROVIDER"),
)

# Seconds before an LLM request is abandoned. For a streamed reply it applies to the
# wait for each chunk, so a long answer is fine but a stalled provider is not.
LLM_TIMEOUT_S: int = 120
# LiteLLM retries for a background call with no fallback model (the summary, the org brief).
BACKGROUND_RETRIES: int = 3


def load_settings(env_file: Path) -> None:
    """
    Load the .env file into the environment (variables already set win) and apply its TZ.

    The entry points call this before importing the rest of agent/. Python reads
    TZ when it starts, so a TZ from .env changed nothing outside Docker (where
    Compose sets it first) until time.tzset() read it again; Windows has no tzset.
    """
    load_dotenv(env_file)
    if not os.environ.get("TZ"):
        os.environ.pop("TZ", None)  # "TZ=" in .env means unset (the server's zone), not UTC
    if hasattr(time, "tzset"):
        time.tzset()


@dataclass(frozen=True)
class _Provider:
    """
    Static descriptor for one LLM provider.

    id:              the name LLM_PROVIDER, FAST_PROVIDER and JUDGE_PROVIDER use.
    key_env:         variable holding the API key, passed as api_key.
    model_env:       variable that overrides default_model.
    default_model:   the model used when model_env is unset.
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
        default_model="claude-haiku-5-5",
        requires=("ANTHROPIC_API_KEY",),
    ),
    _Provider(
        id="openai",
        key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        model_prefixes=("gpt-", "o1", "o3", "o4", "chatgpt-", "openai/"),
        default_model="gpt-4o-mini",
        requires=("OPENAI_API_KEY",),
    ),
    _Provider(
        id="xai",
        key_env="XAI_API_KEY",
        model_env="XAI_MODEL",
        model_prefixes=("xai/",),
        default_model="xai/grok-3-mini",
        requires=("XAI_API_KEY",),
        litellm_prefix="xai/",
    ),
    _Provider(
        id="google",
        key_env="GEMINI_API_KEY",
        model_env="GEMINI_MODEL",
        model_prefixes=("gemini/",),
        default_model="gemini/gemini-2.0-flash",
        requires=("GEMINI_API_KEY",),
        litellm_prefix="gemini/",
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
    _Provider(  # OpenCode Zen — OpenAI-compatible; its free models refuse clients other than OpenCode
        id="opencode",
        key_env="OPENCODE_API_KEY",
        model_env="OPENCODE_MODEL",
        model_prefixes=("openai/",),
        # Cheapest paid Chat Completions model ($0.15 in / $0.50 out per 1M tokens).
        default_model="openai/glm-5.3-flash",
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


def _with_litellm_prefix(model: str, provider: _Provider) -> str:
    """`model` with the provider's LiteLLM prefix in front, unless it already has it."""
    if provider.litellm_prefix and not model.startswith(provider.litellm_prefix):
        return provider.litellm_prefix + model
    return model


def _provider_model(provider: _Provider) -> str:
    """The provider's configured (or default) model, with its LiteLLM prefix added."""
    return _with_litellm_prefix(os.getenv(provider.model_env) or provider.default_model, provider)


def _named_provider(variable: str, configured: list[_Provider]) -> _Provider | None:
    """
    The configured provider whose id the environment variable names, else None.

    An unknown or unconfigured id is reported once and then ignored.
    """
    name = os.getenv(variable, "").strip().lower()
    if not name:
        return None
    provider = next((p for p in configured if p.id == name), None)
    if provider is None:
        _report_unusable_provider(variable, name)
    return provider


@functools.cache
def _report_unusable_provider(variable: str, name: str) -> None:
    """Log, once per variable and value, that it names no configured provider."""
    known = ", ".join(p.id for p in PROVIDERS)
    logger.error(
        "%s=%r is not a configured provider (known: %s; its key variables must be set). "
        "Ignoring it.",
        variable,
        name,
        known,
    )


def _configured_providers() -> list[_Provider]:
    """
    Configured providers, highest priority first.

    LLM_PROVIDER (a provider id) moves that provider to the front; an unknown
    or unconfigured id is logged once and the registry order is used instead.
    """
    configured = [p for p in PROVIDERS if _is_configured(p)]
    chosen = _named_provider("LLM_PROVIDER", configured)
    if chosen is None:
        return configured
    return [chosen] + [p for p in configured if p is not chosen]


def provider_configured() -> bool:
    """Whether any provider is configured; without one, calls go to the Anthropic default unkeyed."""
    return bool(_configured_providers())


def resolve_model() -> str:
    """
    Resolve the LLM model to use from environment variables.

    Returns the model of the highest-priority configured provider (see
    _configured_providers). Logs an error if the configured model doesn't
    match its provider (e.g. ANTHROPIC_API_KEY set but
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
        logger.error(
            "%s=%r does not match provider '%s' (expected prefix in %s). "
            "Check your .env configuration.",
            selected.model_env,
            model,
            selected.id,
            selected.model_prefixes,
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


def _role_model(model_variable: str, provider_variable: str) -> tuple[str, _Provider | None]:
    """
    A background role's model and the provider it names, if any.

    With a provider named, the model gets that provider's LiteLLM prefix, and
    an unset model means the provider's own configured (or default) model.
    Without one, an unset model means the main model.
    """
    provider = _named_provider(provider_variable, _configured_providers())
    model = os.getenv(model_variable, "")
    if provider is None:
        return model or resolve_model(), None
    return (_with_litellm_prefix(model, provider) if model else _provider_model(provider)), provider


def _provider_for(model: str) -> _Provider | None:
    """
    The configured provider that serves `model`.

    In order: the provider whose own model is exactly `model`; the provider
    FAST_PROVIDER or JUDGE_PROVIDER names, when `model` is that role's model;
    the first whose prefixes match. A prefix can belong to several providers:
    "openai/" is OpenAI's and also OpenCode Zen's. With both configured and
    no provider named, openai/deepseek-v4-pro goes to the one higher in
    PROVIDERS (OpenAI), and a warning says so instead of failing silently.
    """
    configured = _configured_providers()
    if exact := next((p for p in configured if _provider_model(p) == model), None):
        return exact
    for variables in _ROLE_VARIABLES:
        role_model, named = _role_model(*variables)
        if named and role_model == model:
            return named
    matching = [p for p in configured if p.model_prefixes and model_matches_provider(model, p)]
    if len(matching) > 1:
        _warn_ambiguous_model(model, tuple(p.id for p in matching))
    return matching[0] if matching else None


@functools.cache
def _warn_ambiguous_model(model: str, provider_ids: tuple[str, ...]) -> None:
    """Log, once per model, that several configured providers could serve it."""
    logger.warning(
        "%r could belong to %s; using %s. To choose, set FAST_PROVIDER or JUDGE_PROVIDER "
        "to the provider's id.",
        model,
        " or ".join(provider_ids),
        provider_ids[0],
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


def connection_kwargs(model: str) -> dict[str, Any]:
    """
    Return the timeout, api_base and api_key to pass to litellm.acompletion() with model.

    Every call gets LLM_TIMEOUT_S, so a provider that hangs ends in an error
    (and the next provider, for chat replies) instead of a UI that waits forever.

    The provider is the one _provider_for() picks: the configured one whose
    model is exactly `model`, the one FAST_PROVIDER or JUDGE_PROVIDER names for
    that role's model, else the first whose prefixes match. Native providers need nothing
    here when their key uses LiteLLM's own variable name, but passing it is
    harmless; OpenAI-compatible endpoints (OpenCode Zen, vLLM, LLM_API_BASE)
    need both values, since LiteLLM cannot guess them. Unknown models get only
    the timeout, and LiteLLM reads its usual environment variables.

    Args:
        model: A LiteLLM model string, e.g. from resolve_model() or FAST_MODEL.

    Returns:
        Dict with "timeout", plus "api_base" and/or "api_key" for values that are set.
    """
    provider = _provider_for(model)
    if provider is None:
        return {"timeout": LLM_TIMEOUT_S}
    settings = {
        "api_base": (os.getenv(provider.base_url_env) if provider.base_url_env else "")
        or provider.default_base_url,
        "api_key": os.getenv(provider.key_env, ""),
    }
    return {"timeout": LLM_TIMEOUT_S, **{name: value for name, value in settings.items() if value}}


def resolve_fast_model() -> str:
    """
    Return the model for background work: summaries, org briefs, calendar plans.

    FAST_MODEL lets a deployment route that work to a cheaper model (cost
    routing) while user-facing drafts keep the main model, and FAST_PROVIDER
    names its provider (see _role_model). With neither set, it is the same as
    resolve_model(). The provider must be configured.
    """
    return _role_model("FAST_MODEL", "FAST_PROVIDER")[0]


def resolve_judge_model() -> str:
    """
    Return the model that grades scripts/eval.py runs.

    JUDGE_MODEL and JUDGE_PROVIDER work like FAST_MODEL and FAST_PROVIDER;
    with neither set, the judge is the main model.
    """
    return _role_model("JUDGE_MODEL", "JUDGE_PROVIDER")[0]
