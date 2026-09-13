from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse


_PROVIDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_ALLOWED_PROTOCOLS = frozenset({"openai", "gemini"})


@dataclass(frozen=True)
class AIProvider:
    id: str
    label: str
    protocol: str
    api_base: str
    default_qc_model: str
    default_translation_model: str | None
    env_names: tuple[str, ...]
    builtin: bool = True

    @property
    def chat_url(self) -> str | None:
        if self.protocol != "openai":
            return None
        return f"{self.api_base.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.api_base.rstrip('/')}/models"


PROVIDERS: dict[str, AIProvider] = {
    "gemini": AIProvider(
        id="gemini",
        label="Google Gemini",
        protocol="gemini",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        default_qc_model=os.getenv("GEMINI_VISUAL_QC_MODEL", "gemini-3.7-flash"),
        default_translation_model=None,
        env_names=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    "deepseek": AIProvider(
        id="deepseek",
        label="DeepSeek",
        protocol="openai",
        api_base="https://api.deepseek.com",
        default_qc_model=os.getenv("DEEPSEEK_VISUAL_QC_MODEL", "deepseek-v4-flash-vision-exp"),
        default_translation_model=os.getenv("MANGA_DEEPSEEK_TRANSLATION_MODEL", "deepseek-v4-flash"),
        env_names=("DEEPSEEK_API_KEY",),
    ),
    "openai": AIProvider(
        id="openai",
        label="OpenAI",
        protocol="openai",
        api_base="https://api.openai.com/v1",
        default_qc_model="gpt-5-mini",
        default_translation_model="gpt-5-mini",
        env_names=("OPENAI_API_KEY",),
    ),
    "openrouter": AIProvider(
        id="openrouter",
        label="OpenRouter",
        protocol="openai",
        api_base="https://openrouter.ai/api/v1",
        default_qc_model="google/gemini-2.5-flash",
        default_translation_model="google/gemini-2.5-flash",
        env_names=("OPENROUTER_API_KEY",),
    ),
    "experiential": AIProvider(
        id="experiential",
        label="Experiential Labs",
        protocol="openai",
        api_base="https://api.experientiallabs.ai/v1",
        default_qc_model="deepseek-v4.1-flash",
        default_translation_model="deepseek-v4.1-flash",
        env_names=("EXPLABS_API_KEY",),
    ),
}

PROVIDER_IDS = frozenset(PROVIDERS)


def normalize_provider_id(value: str) -> str:
    provider_id = (value or "").strip().lower()
    if not _PROVIDER_ID_RE.fullmatch(provider_id):
        raise ValueError(
            "AI provider id must be 1-64 lowercase letters, numbers, '-' or '_'"
        )
    return provider_id


def validate_provider_label(value: str | None, *, default: str = "Custom AI") -> str:
    label = (value or default).strip()
    if not label or len(label) > 80:
        raise ValueError("AI provider label must contain between 1 and 80 characters")
    if any(ord(char) < 32 for char in label):
        raise ValueError("AI provider label contains invalid control characters")
    return label


def validate_protocol(value: str | None) -> str:
    protocol = (value or "").strip().lower()
    if protocol not in _ALLOWED_PROTOCOLS:
        raise ValueError("AI protocol must be 'openai' or 'gemini'")
    return protocol


def validate_api_base(value: str | None) -> str:
    api_base = (value or "").strip().rstrip("/")
    if not api_base or len(api_base) > 2048:
        raise ValueError("AI API base URL is required")
    try:
        parsed = urlparse(api_base)
    except ValueError as exc:
        raise ValueError("AI API base URL is invalid") from exc
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("AI API base URL must be a public HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials are not allowed inside the AI API URL")
    if parsed.query or parsed.fragment:
        raise ValueError("AI API base URL must not contain a query or fragment")
    return api_base


def get_provider(provider_id: str) -> AIProvider:
    provider = PROVIDERS.get(normalize_provider_id(provider_id))
    if provider is None:
        raise ValueError(f"Unsupported AI provider: {provider_id}")
    return provider


def resolve_provider(
    provider_id: str,
    *,
    label: str | None = None,
    protocol: str | None = None,
    api_base: str | None = None,
) -> AIProvider:
    """Resolve a built-in provider or a user-defined OpenAI-compatible endpoint.

    Built-in provider IDs always use their audited endpoint and protocol. Custom
    providers intentionally support the OpenAI-compatible contract only; the
    Gemini path uses application-specific interactions payloads and therefore
    remains a built-in integration instead of pretending arbitrary Gemini-like
    endpoints are compatible.
    """

    normalized = normalize_provider_id(provider_id)
    builtin = PROVIDERS.get(normalized)
    if builtin is not None:
        return builtin

    resolved_protocol = validate_protocol(protocol)
    if resolved_protocol != "openai":
        raise ValueError(
            "Custom providers must use the OpenAI-compatible protocol"
        )
    return AIProvider(
        id=normalized,
        label=validate_provider_label(label),
        protocol=resolved_protocol,
        api_base=validate_api_base(api_base),
        default_qc_model="",
        default_translation_model=None,
        env_names=(),
        builtin=False,
    )


def validate_model_name(value: str | None, *, default: str) -> str:
    model = (value or default).strip()
    if not model or len(model) > 200:
        raise ValueError("AI model name must contain between 1 and 200 characters")
    if any(ord(char) < 32 for char in model):
        raise ValueError("AI model name contains invalid control characters")
    return model
