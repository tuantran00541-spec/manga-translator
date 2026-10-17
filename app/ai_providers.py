from __future__ import annotations

import os
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse


AIProtocol = Literal["gemini", "openai"]
AIRequestProfile = Literal["gemini", "standard", "deepseek"]
AIImageTransport = Literal["inline_base64", "data_url"]

_PROVIDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_ALLOWED_PROTOCOLS = frozenset({"openai", "gemini"})


@dataclass(frozen=True)
class AIProvider:
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
    builtin: bool = True

    @property
    def chat_url(self) -> str | None:
        if self.protocol != "openai":
            return None
        return f"{self.api_base.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.api_base.rstrip('/')}/models"

    def chat_completion_extras(self) -> dict:
        if self.protocol != "openai":
            return {}
        if self.request_profile == "deepseek":
            return {"thinking": {"type": "disabled"}}
        return {}

    def public_capabilities(self) -> dict:
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
}

PROVIDER_IDS = frozenset(PROVIDERS)

CLOUD_PROVIDER_ID = "manga-cloud"
_cloud_job: ContextVar[tuple[AIProvider, str] | None] = ContextVar("cloud_job", default=None)


def bind_cloud_job(provider: AIProvider, job_token: str):
    return _cloud_job.set((provider, job_token))


def unbind_cloud_job(token) -> None:
    _cloud_job.reset(token)


def cloud_job_provider(provider_id: str) -> AIProvider | None:
    if provider_id != CLOUD_PROVIDER_ID:
        return None
    bound = _cloud_job.get()
    if bound is None:
        raise ValueError("Manga Cloud chỉ dùng được bên trong A.I mode")
    return bound[0]


def cloud_job_key() -> str | None:
    bound = _cloud_job.get()
    return bound[1] if bound else None


def normalize_provider_id(value: str) -> str:
    provider_id = (value or "").strip().lower()
    if not _PROVIDER_ID_RE.fullmatch(provider_id):
        raise ValueError(
            "AI provider id must be 1-64 lowercase letters, numbers, '-' or '_'"
        )
    return provider_id


def validate_provider_label(
    value: str | None,
    *,
    default: str = "Custom AI",
) -> str:
    label = (value or default).strip()
    if not label or len(label) > 80:
        raise ValueError(
            "AI provider label must contain between 1 and 80 characters"
        )
    if any(ord(char) < 32 for char in label):
        raise ValueError("AI provider label contains invalid control characters")
    return label


def validate_protocol(value: str | None) -> str:
    protocol = (value or "").strip().lower()
    if protocol not in _ALLOWED_PROTOCOLS:
        raise ValueError("AI protocol must be 'openai' or 'gemini'")
    return protocol


# Where each well-known host serves its OpenAI-compatible API, so a bare host or a pasted endpoint still works.
KNOWN_API_BASES: dict[str, str] = {
    "api.openai.com": "/v1",
    "api.anthropic.com": "/v1",
    "generativelanguage.googleapis.com": "/v1beta/openai",
    "api.deepseek.com": "/v1",
    "openrouter.ai": "/api/v1",
    "api.groq.com": "/openai/v1",
    "api.together.xyz": "/v1",
    "api.fireworks.ai": "/inference/v1",
    "api.mistral.ai": "/v1",
    "api.x.ai": "/v1",
    "api.cerebras.ai": "/v1",
    "api.sambanova.ai": "/v1",
    "api.perplexity.ai": "",
    "api.moonshot.ai": "/v1",
    "api.moonshot.cn": "/v1",
    "api.z.ai": "/api/paas/v4",
    "open.bigmodel.cn": "/api/paas/v4",
    "dashscope.aliyuncs.com": "/compatible-mode/v1",
    "dashscope-intl.aliyuncs.com": "/compatible-mode/v1",
    "api.siliconflow.cn": "/v1",
    "api.siliconflow.com": "/v1",
    "integrate.api.nvidia.com": "/v1",
    "router.huggingface.co": "/v1",
    "models.github.ai": "/inference",
    "apihub.agnes-ai.com": "/v1",
}
_AZURE_SUFFIXES = (".openai.azure.com", ".services.ai.azure.com", ".cognitiveservices.azure.com")
_ENDPOINT_TAILS = ("/chat/completions", "/completions", "/responses", "/embeddings", "/messages", "/models")
_VERSION_TAIL = re.compile(r"/v\d+[a-z0-9]*$")


def is_azure_host(host: str | None) -> bool:
    return bool(host) and str(host).lower().endswith(_AZURE_SUFFIXES)


def normalize_api_base(value: str | None) -> str:
    """The API root from whatever the user pasted: a bare host, a docs URL or a full endpoint."""
    raw = (value or "").strip()
    if raw and "://" not in raw:
        raw = "https://" + raw
    try:
        parsed = urlparse(raw)
    except ValueError:
        return raw.rstrip("/")
    if not parsed.hostname:
        return raw.rstrip("/")
    host = parsed.hostname.lower()
    path = re.sub(r"/{2,}", "/", parsed.path or "").rstrip("/")
    lowered = path.lower()
    for tail in _ENDPOINT_TAILS:
        if lowered.endswith(tail):
            path, lowered = path[: -len(tail)].rstrip("/"), lowered[: -len(tail)].rstrip("/")
            break
    known = KNOWN_API_BASES.get(host)
    if known is not None and (not path or known.lower().startswith(lowered + "/")):
        path = known
    elif is_azure_host(host) and not lowered.endswith("/openai/v1"):
        # Azure's v1 API needs no api-version; deployment URLs map onto it.
        path = "/openai/v1"
    scheme = (parsed.scheme or "https").lower()
    netloc = host if parsed.port in (None, 443) else f"{host}:{parsed.port}"
    return f"{scheme}://{netloc}{path}"


def api_base_candidates(api_base: str) -> list[str]:
    """Roots to try, in order, when a host is not one we know."""
    base = api_base.rstrip("/")
    host = (urlparse(base).hostname or "").lower()
    if host in KNOWN_API_BASES or is_azure_host(host) or _VERSION_TAIL.search(base):
        return [base]
    return [base, f"{base}/v1", f"{base}/api/v1", f"{base}/openai/v1"]


def auth_headers(provider: "AIProvider", api_key: str) -> dict:
    """Bearer auth, plus the api-key header Azure expects."""
    key = (api_key or "").strip()
    headers = {"Authorization": f"Bearer {key}"}
    if is_azure_host(urlparse(provider.api_base).hostname):
        headers["api-key"] = key
    return headers


def validate_api_base(value: str | None) -> str:
    try:
        pasted = urlparse((value or "").strip() if "://" in (value or "") else "https://" + (value or "").strip())
        if pasted.username is not None or pasted.password is not None:
            raise ValueError("Credentials are not allowed inside the AI API URL")
    except ValueError as exc:
        raise ValueError(str(exc) if "Credentials" in str(exc) else "AI API base URL is invalid") from exc
    api_base = normalize_api_base(value)
    if not api_base or len(api_base) > 2048:
        raise ValueError("AI API base URL is required")
    try:
        parsed = urlparse(api_base)
    except ValueError as exc:
        raise ValueError("AI API base URL is invalid") from exc
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("AI API base URL must be a public HTTPS URL")
    return api_base


def resolve_provider(
    provider_id: str,
    *,
    label: str | None = None,
    protocol: str | None = None,
    api_base: str | None = None,
) -> AIProvider:
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
        label=validate_provider_label(label, default=normalized),
        protocol="openai",
        api_base=validate_api_base(api_base),
        default_qc_model="",
        default_translation_model="",
        env_names=(),
        request_profile="standard",
        image_transport="data_url",
        supports_visual_qc=True,
        supports_translation=True,
        tracks_cost=False,
        builtin=False,
    )


def validate_model_name(value: str | None, *, default: str) -> str:
    model = (value or default).strip()
    if not model or len(model) > 200:
        raise ValueError("AI model name must contain between 1 and 200 characters")
    if any(ord(char) < 32 for char in model):
        raise ValueError("AI model name contains invalid control characters")
    return model
