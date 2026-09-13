from __future__ import annotations

import json
import os

from app.ai_providers import PROVIDERS, normalize_provider_id, validate_provider_label

_SERVICE_NAME = "manga-translator"
_GEMINI_ACCOUNT = "gemini-api-key"
_DEEPSEEK_ACCOUNT = "deepseek-api-key"
_PROVIDER_CONFIG_INDEX_ACCOUNT = "ai-provider-config-index"


def _provider_account(provider_id: str) -> str:
    return f"ai-provider-{normalize_provider_id(provider_id)}-api-key"


def _provider_config_account(provider_id: str) -> str:
    return f"ai-provider-{normalize_provider_id(provider_id)}-config"


def _provider_meta(provider_id: str, provider_label: str | None = None) -> tuple[str, tuple[str, ...], str]:
    normalized = normalize_provider_id(provider_id)
    provider = PROVIDERS.get(normalized)
    if provider is not None:
        return provider.id, provider.env_names, provider.label
    return normalized, (), validate_provider_label(provider_label, default=normalized)


class SecretStoreUnavailable(RuntimeError):
    pass


def _keyring_module():
    try:
        import keyring  # type: ignore
        from keyring.errors import KeyringError  # type: ignore
    except ImportError as exc:
        raise SecretStoreUnavailable(
            "Secure secret storage is unavailable. Install the 'keyring' dependency."
        ) from exc
    return keyring, KeyringError


def _get_api_key(account: str, env_names: tuple[str, ...], provider: str) -> str | None:
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return env_value.strip()

    keyring, KeyringError = _keyring_module()
    try:
        value = keyring.get_password(_SERVICE_NAME, account)
    except KeyringError as exc:
        raise SecretStoreUnavailable(
            f"Cannot read {provider} API key from OS secure storage: {exc}"
        ) from exc
    return value.strip() if value and value.strip() else None


def _set_api_key(account: str, value: str, provider: str) -> None:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"{provider} API key is required")
    if len(value) > 4096:
        raise ValueError(f"{provider} API key is unexpectedly long")

    keyring, KeyringError = _keyring_module()
    try:
        keyring.set_password(_SERVICE_NAME, account, value)
    except KeyringError as exc:
        raise SecretStoreUnavailable(
            f"Cannot save {provider} API key to OS secure storage: {exc}"
        ) from exc


def _delete_api_key(account: str, provider: str) -> None:
    keyring, KeyringError = _keyring_module()
    try:
        keyring.delete_password(_SERVICE_NAME, account)
    except KeyringError as exc:
        detail = str(exc).lower()
        if "not found" not in detail and "no password" not in detail:
            raise SecretStoreUnavailable(
                f"Cannot delete {provider} API key from OS secure storage: {exc}"
            ) from exc


def _key_status(
    account: str,
    env_names: tuple[str, ...],
    provider: str,
) -> dict:
    if any(os.getenv(name) for name in env_names):
        return {"configured": True, "source": "environment"}
    try:
        configured = bool(_get_api_key(account, env_names, provider))
        return {
            "configured": configured,
            "source": "os_secure_storage" if configured else "none",
        }
    except SecretStoreUnavailable as exc:
        return {
            "configured": False,
            "source": "unavailable",
            "detail": str(exc),
        }


def get_gemini_api_key() -> str | None:
    return _get_api_key(
        _GEMINI_ACCOUNT,
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "Gemini",
    )


def set_gemini_api_key(value: str) -> None:
    _set_api_key(_GEMINI_ACCOUNT, value, "Gemini")


def delete_gemini_api_key() -> None:
    _delete_api_key(_GEMINI_ACCOUNT, "Gemini")


def gemini_key_status() -> dict:
    return _key_status(
        _GEMINI_ACCOUNT,
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "Gemini",
    )


def get_deepseek_api_key() -> str | None:
    return _get_api_key(
        _DEEPSEEK_ACCOUNT,
        ("DEEPSEEK_API_KEY",),
        "DeepSeek",
    )


def set_deepseek_api_key(value: str) -> None:
    _set_api_key(_DEEPSEEK_ACCOUNT, value, "DeepSeek")


def delete_deepseek_api_key() -> None:
    _delete_api_key(_DEEPSEEK_ACCOUNT, "DeepSeek")


def deepseek_key_status() -> dict:
    return _key_status(
        _DEEPSEEK_ACCOUNT,
        ("DEEPSEEK_API_KEY",),
        "DeepSeek",
    )


def get_provider_api_key(provider_id: str, *, provider_label: str | None = None) -> str | None:
    provider_id, env_names, label = _provider_meta(provider_id, provider_label)
    if provider_id == "gemini":
        return get_gemini_api_key()
    if provider_id == "deepseek":
        return get_deepseek_api_key()
    return _get_api_key(_provider_account(provider_id), env_names, label)


def set_provider_api_key(provider_id: str, value: str, *, provider_label: str | None = None) -> None:
    provider_id, _env_names, label = _provider_meta(provider_id, provider_label)
    account = {
        "gemini": _GEMINI_ACCOUNT,
        "deepseek": _DEEPSEEK_ACCOUNT,
    }.get(provider_id, _provider_account(provider_id))
    _set_api_key(account, value, label)


def delete_provider_api_key(provider_id: str, *, provider_label: str | None = None) -> None:
    provider_id, _env_names, label = _provider_meta(provider_id, provider_label)
    account = {
        "gemini": _GEMINI_ACCOUNT,
        "deepseek": _DEEPSEEK_ACCOUNT,
    }.get(provider_id, _provider_account(provider_id))
    _delete_api_key(account, label)


def provider_key_status(provider_id: str, *, provider_label: str | None = None) -> dict:
    provider_id, env_names, label = _provider_meta(provider_id, provider_label)
    if provider_id == "gemini":
        return gemini_key_status()
    if provider_id == "deepseek":
        return deepseek_key_status()
    return _key_status(_provider_account(provider_id), env_names, label)


def _custom_provider_ids() -> list[str]:
    keyring, KeyringError = _keyring_module()
    try:
        raw = keyring.get_password(_SERVICE_NAME, _PROVIDER_CONFIG_INDEX_ACCOUNT)
    except KeyringError as exc:
        raise SecretStoreUnavailable(
            f"Cannot read AI provider registry from OS secure storage: {exc}"
        ) from exc
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise SecretStoreUnavailable("Stored AI provider registry is invalid") from exc
    if not isinstance(values, list):
        raise SecretStoreUnavailable("Stored AI provider registry is invalid")
    result: list[str] = []
    for value in values:
        try:
            provider_id = normalize_provider_id(str(value))
        except ValueError:
            continue
        if provider_id not in PROVIDERS and provider_id not in result:
            result.append(provider_id)
    return result[:100]


def _set_custom_provider_ids(values: list[str]) -> None:
    normalized = []
    for value in values:
        provider_id = normalize_provider_id(value)
        if provider_id not in PROVIDERS and provider_id not in normalized:
            normalized.append(provider_id)
    keyring, KeyringError = _keyring_module()
    try:
        keyring.set_password(
            _SERVICE_NAME,
            _PROVIDER_CONFIG_INDEX_ACCOUNT,
            json.dumps(normalized[:100], separators=(",", ":")),
        )
    except KeyringError as exc:
        raise SecretStoreUnavailable(
            f"Cannot save AI provider registry to OS secure storage: {exc}"
        ) from exc


def set_provider_config(
    provider_id: str,
    *,
    label: str,
    protocol: str,
    api_base: str,
) -> None:
    normalized = normalize_provider_id(provider_id)
    if normalized in PROVIDERS:
        return
    payload = json.dumps(
        {
            "id": normalized,
            "label": validate_provider_label(label, default=normalized),
            "protocol": str(protocol or "").strip().lower(),
            "api_base": str(api_base or "").strip(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    keyring, KeyringError = _keyring_module()
    try:
        keyring.set_password(_SERVICE_NAME, _provider_config_account(normalized), payload)
    except KeyringError as exc:
        raise SecretStoreUnavailable(
            f"Cannot save {label} provider configuration to OS secure storage: {exc}"
        ) from exc
    ids = _custom_provider_ids()
    if normalized not in ids:
        ids.append(normalized)
        _set_custom_provider_ids(ids)


def get_provider_config(provider_id: str) -> dict | None:
    normalized = normalize_provider_id(provider_id)
    if normalized in PROVIDERS:
        provider = PROVIDERS[normalized]
        return {
            "id": provider.id,
            "label": provider.label,
            "protocol": provider.protocol,
            "api_base": provider.api_base,
        }
    keyring, KeyringError = _keyring_module()
    try:
        raw = keyring.get_password(_SERVICE_NAME, _provider_config_account(normalized))
    except KeyringError as exc:
        raise SecretStoreUnavailable(
            f"Cannot read provider configuration from OS secure storage: {exc}"
        ) from exc
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise SecretStoreUnavailable("Stored AI provider configuration is invalid") from exc
    return payload if isinstance(payload, dict) else None


def list_provider_configs() -> list[dict]:
    result: list[dict] = []
    for provider_id in _custom_provider_ids():
        config = get_provider_config(provider_id)
        if isinstance(config, dict):
            result.append(config)
    return result


def delete_provider_config(provider_id: str) -> None:
    normalized = normalize_provider_id(provider_id)
    if normalized in PROVIDERS:
        return
    keyring, KeyringError = _keyring_module()
    try:
        keyring.delete_password(_SERVICE_NAME, _provider_config_account(normalized))
    except KeyringError as exc:
        detail = str(exc).lower()
        if "not found" not in detail and "no password" not in detail:
            raise SecretStoreUnavailable(
                f"Cannot delete AI provider configuration: {exc}"
            ) from exc
    ids = [provider_id for provider_id in _custom_provider_ids() if provider_id != normalized]
    _set_custom_provider_ids(ids)
