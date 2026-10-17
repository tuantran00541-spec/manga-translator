"""A small JSON Schema check and a JSON extractor, so a helper's report can be required to have a shape (from dsh's workflow agent())."""
from __future__ import annotations

import json

TYPES = {"string": str, "boolean": bool, "array": list, "object": dict}


def extract(text: str):
    """The last top-level JSON object in the text, or None."""
    decoder, found, at = json.JSONDecoder(), None, 0
    while (start := text.find("{", at)) != -1:
        try:
            found, used = decoder.raw_decode(text[start:])
            at = start + used
        except ValueError:
            at = start + 1
    return found


def check(value, schema: dict, path: str = "report") -> str | None:
    """The first way the value breaks the schema (type, required, properties, items, enum, const), or None."""
    if "const" in schema and value != schema["const"]:
        return f"{path} must be {schema['const']!r}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']}"
    kind = schema.get("type")
    if kind == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        ok = kind is None or isinstance(value, TYPES.get(kind, object))
    if not ok:
        return f"{path} must be a {kind}"
    if isinstance(value, dict):
        for name in schema.get("required") or []:
            if name not in value:
                return f"{path}.{name} is missing"
        for name, sub in (schema.get("properties") or {}).items():
            if name in value and isinstance(sub, dict) and (error := check(value[name], sub, f"{path}.{name}")):
                return error
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            if error := check(item, schema["items"], f"{path}[{i}]"):
                return error
    return None


def report(text: str, schema: dict) -> tuple[dict | None, str | None]:
    """(object, None) when the text ends with a JSON object that fits the schema, else (None, why)."""
    value = extract(text)
    if value is None:
        return None, "the reply holds no JSON object"
    error = check(value, schema)
    return (None, error) if error else (value, None)
