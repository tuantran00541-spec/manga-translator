from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AIProvider:
    id: str
    label: str
    protocol: str
    api_base: str
    default_qc_model: str
    default_translation_model: str | None
    env_names: tuple[str, ...]

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


def get_provider(provider_id: str) -> AIProvider:
    provider = PROVIDERS.get((provider_id or "").strip().lower())
    if provider is None:
        raise ValueError(f"Unsupported AI provider: {provider_id}")
    return provider


def validate_model_name(value: str | None, *, default: str) -> str:
    model = (value or default).strip()
    if not model or len(model) > 200:
        raise ValueError("AI model name must contain between 1 and 200 characters")
    if any(ord(char) < 32 for char in model):
        raise ValueError("AI model name contains invalid control characters")
    return model
