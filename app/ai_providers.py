from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


AIProtocol = Literal["gemini", "openai"]
AIRequestProfile = Literal["gemini", "standard", "deepseek"]
AIImageTransport = Literal["inline_base64", "data_url"]


@dataclass(frozen=True)
class AIProvider:
    """Server-owned contract for one AI provider.

    The UI only selects a provider/model. Authentication, endpoint selection,
    image transport, provider-specific request knobs and cost behaviour stay on
    the backend so browser code does not need provider-specific HTTP logic.
    """

    id: str
    label: str
    protocol: AIProtocol
    api_base: str
    default_qc_model: str
    default_translation_model: str | None
    env_names: tuple[str, ...]
    request_profile: AIRequestProfile
    image_transport: AIImageTransport
    supports_visual_qc: bool = True
    supports_translation: bool = False
    tracks_cost: bool = False

    @property
    def chat_url(self) -> str | None:
        if self.protocol != "openai":
            return None
        return f"{self.api_base.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.api_base.rstrip('/')}/models"

    def chat_completion_extras(self) -> dict:
        """Return provider-specific OpenAI-compatible body fields.

        DeepSeek accepts the explicit thinking control used by this app. Other
        OpenAI-compatible providers must not receive that vendor-specific field.
        """

        if self.protocol != "openai":
            return {}
        if self.request_profile == "deepseek":
            return {"thinking": {"type": "disabled"}}
        return {}

    def public_capabilities(self) -> dict:
        """Stable metadata that mirrors the controls exposed by the current UI."""

        return {
            "visual_qc": self.supports_visual_qc,
            "translation": self.supports_translation,
            "model_listing": True,
            "cost_tracking": self.tracks_cost,
            "image_transport": self.image_transport,
        }


PROVIDERS: dict[str, AIProvider] = {
    "gemini": AIProvider(
        id="gemini",
        label="Google Gemini",
        protocol="gemini",
        api_base="https://generativelanguage.googleapis.com/v1beta",
        default_qc_model=os.getenv("GEMINI_VISUAL_QC_MODEL", "gemini-3.7-flash"),
        default_translation_model=None,
        env_names=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        request_profile="gemini",
        image_transport="inline_base64",
        supports_visual_qc=True,
        supports_translation=False,
        tracks_cost=False,
    ),
    "deepseek": AIProvider(
        id="deepseek",
        label="DeepSeek",
        protocol="openai",
        api_base="https://api.deepseek.com",
        default_qc_model=os.getenv("DEEPSEEK_VISUAL_QC_MODEL", "deepseek-v4-flash-vision-exp"),
        default_translation_model=os.getenv("MANGA_DEEPSEEK_TRANSLATION_MODEL", "deepseek-v4-flash"),
        env_names=("DEEPSEEK_API_KEY",),
        request_profile="deepseek",
        image_transport="data_url",
        supports_visual_qc=True,
        supports_translation=True,
        tracks_cost=True,
    ),
    "openai": AIProvider(
        id="openai",
        label="OpenAI",
        protocol="openai",
        api_base="https://api.openai.com/v1",
        default_qc_model="gpt-5-mini",
        default_translation_model="gpt-5-mini",
        env_names=("OPENAI_API_KEY",),
        request_profile="standard",
        image_transport="data_url",
        supports_visual_qc=True,
        supports_translation=True,
        tracks_cost=False,
    ),
    "openrouter": AIProvider(
        id="openrouter",
        label="OpenRouter",
        protocol="openai",
        api_base="https://openrouter.ai/api/v1",
        default_qc_model="google/gemini-2.5-flash",
        default_translation_model="google/gemini-2.5-flash",
        env_names=("OPENROUTER_API_KEY",),
        request_profile="standard",
        image_transport="data_url",
        supports_visual_qc=True,
        supports_translation=True,
        tracks_cost=False,
    ),
    "experiential": AIProvider(
        id="experiential",
        label="Experiential Labs",
        protocol="openai",
        api_base="https://api.experientiallabs.ai/v1",
        default_qc_model="deepseek-v4.1-flash",
        default_translation_model="deepseek-v4.1-flash",
        env_names=("EXPLABS_API_KEY",),
        request_profile="standard",
        image_transport="data_url",
        supports_visual_qc=True,
        supports_translation=True,
        tracks_cost=False,
    ),
}

PROVIDER_IDS = frozenset(PROVIDERS)


def get_provider(provider_id: str) -> AIProvider:
    provider = PROVIDERS.get((provider_id or "").strip().lower())
    if provider is None:
        raise ValueError(f"Unsupported AI provider: {provider_id}")
    return provider


def get_translation_provider(provider_id: str) -> AIProvider:
    provider = get_provider(provider_id)
    if (
        provider.protocol != "openai"
        or not provider.supports_translation
        or not provider.default_translation_model
    ):
        raise ValueError(f"{provider.label} is not available for translation")
    return provider


def validate_model_name(value: str | None, *, default: str) -> str:
    model = (value or default).strip()
    if not model or len(model) > 200:
        raise ValueError("AI model name must contain between 1 and 200 characters")
    if any(ord(char) < 32 for char in model):
        raise ValueError("AI model name contains invalid control characters")
    return model
