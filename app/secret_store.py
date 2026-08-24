from __future__ import annotations

import os

_SERVICE_NAME = "manga-translator"
_GEMINI_ACCOUNT = "gemini-api-key"


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


def get_gemini_api_key() -> str | None:
    env_value = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_value and env_value.strip():
        return env_value.strip()

    keyring, KeyringError = _keyring_module()
    try:
        value = keyring.get_password(_SERVICE_NAME, _GEMINI_ACCOUNT)
    except KeyringError as exc:
        raise SecretStoreUnavailable(f"Cannot read Gemini API key from OS secure storage: {exc}") from exc
    return value.strip() if value and value.strip() else None


def set_gemini_api_key(value: str) -> None:
    value = (value or "").strip()
    if not value:
        raise ValueError("Gemini API key is required")
    if len(value) > 4096:
        raise ValueError("Gemini API key is unexpectedly long")

    keyring, KeyringError = _keyring_module()
    try:
        keyring.set_password(_SERVICE_NAME, _GEMINI_ACCOUNT, value)
    except KeyringError as exc:
        raise SecretStoreUnavailable(f"Cannot save Gemini API key to OS secure storage: {exc}") from exc


def delete_gemini_api_key() -> None:
    keyring, KeyringError = _keyring_module()
    try:
        keyring.delete_password(_SERVICE_NAME, _GEMINI_ACCOUNT)
    except KeyringError as exc:
        # Different backends use backend-specific exceptions for a missing entry.
        if "not found" not in str(exc).lower() and "no password" not in str(exc).lower():
            raise SecretStoreUnavailable(f"Cannot delete Gemini API key from OS secure storage: {exc}") from exc


def gemini_key_status() -> dict:
    if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return {"configured": True, "source": "environment"}
    try:
        configured = bool(get_gemini_api_key())
        return {"configured": configured, "source": "os_secure_storage" if configured else "none"}
    except SecretStoreUnavailable as exc:
        return {"configured": False, "source": "unavailable", "detail": str(exc)}
