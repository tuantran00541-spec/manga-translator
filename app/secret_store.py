from __future__ import annotations

import os

_SERVICE_NAME = "manga-translator"
_GEMINI_ACCOUNT = "gemini-api-key"
_DEEPSEEK_ACCOUNT = "deepseek-api-key"


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
