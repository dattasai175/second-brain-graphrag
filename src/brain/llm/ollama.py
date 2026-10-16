"""Ollama-backed LLM client (local-first). Uses the native /api/chat structured-output support."""

from __future__ import annotations

import json

import httpx
from pydantic import ValidationError

from ..config import LLMSettings
from .base import LLMOutputError, LLMUnavailable, T


class OllamaClient:
    def __init__(self, cfg: LLMSettings):
        self.cfg = cfg
        self.model = cfg.model
        self.base_url = cfg.base_url.rstrip("/")
        self._http = httpx.Client(timeout=cfg.timeout_s)

    # --------------------------------------------------------------- plumbing
    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = self._http.post(f"{self.base_url}{path}", json=payload)
        except httpx.ConnectError as e:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve` "
                "(or the Ollama app)."
            ) from e
        except httpx.TimeoutException as e:
            raise LLMUnavailable(
                f"Ollama request to {path} timed out after {self.cfg.timeout_s}s"
            ) from e
        if resp.status_code == 404:
            model = payload.get("model", "?")
            raise LLMUnavailable(f"Ollama model '{model}' not found. Run: ollama pull {model}")
        if resp.status_code >= 400:
            raise LLMUnavailable(f"Ollama {path} failed ({resp.status_code}): {resp.text[:300]}")
        return resp.json()

    def list_models(self) -> list[str]:
        try:
            data = self._http.get(f"{self.base_url}/api/tags").json()
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"Cannot reach Ollama at {self.base_url}") from e
        return [m["name"] for m in data.get("models", [])]

    def has_model(self, name: str) -> bool:
        names = self.list_models()
        return name in names or f"{name}:latest" in names

    # ------------------------------------------------------------------- chat
    def _chat(
        self,
        system: str,
        user: str,
        *,
        model: str | None,
        schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> str:
        options: dict = {"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx}
        if max_tokens:
            options["num_predict"] = max_tokens
        payload: dict = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options,
        }
        if schema is not None:
            payload["format"] = schema
        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    def chat(
        self, system: str, user: str, *, model: str | None = None, max_tokens: int | None = None
    ) -> str:
        return self._chat(system, user, model=model, max_tokens=max_tokens).strip()

    def structured(self, system: str, user: str, schema: type[T], *, model: str | None = None) -> T:
        json_schema = schema.model_json_schema()
        prompt = user
        last_err: Exception | None = None
        for _ in range(2):
            raw = self._chat(system, prompt, model=model, schema=json_schema)
            try:
                return schema.model_validate_json(raw)
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                last_err = e
                prompt = (
                    f"{user}\n\nYour previous reply was not valid for the required JSON schema "
                    f"({str(e)[:200]}). Reply again with only valid JSON."
                )
        raise LLMOutputError(f"Model returned invalid structured output: {last_err}")
