"""Thin HTTP clients for remote / local model servers.

OpenAIClient works with anything that speaks the OpenAI API: OpenAI, OpenRouter, Groq, Together, Mistral,
Gemini (OpenAI-compatible endpoint), DeepSeek, and local servers (llama.cpp `llama-server`, Ollama, vLLM,
LM Studio). AnthropicClient talks to the Anthropic Messages API.
"""
from __future__ import annotations

import json
import re

import httpx


def _json_from(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)  # models sometimes wrap JSON in prose or ```json fences
        if not m:
            raise
        return json.loads(m.group(0))


class OpenAIClient:
    def __init__(self, base_url: str, model: str, api_key: str | None, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.model = model
        h = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(timeout=timeout, headers=h)

    def chat_json(self, system: str, user: str, schema: dict | None = None, max_tokens: int = 200) -> dict:
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "result", "schema": schema}}
        r = self.http.post(self.base + "/chat/completions", json=body)
        if r.status_code == 400 and schema:  # server without structured outputs: fall back to plain JSON mode
            body["response_format"] = {"type": "json_object"}
            r = self.http.post(self.base + "/chat/completions", json=body)
        r.raise_for_status()
        return _json_from(r.json()["choices"][0]["message"]["content"])

    def vision_json(self, prompt: str, jpeg_b64: str) -> dict:
        body = {"model": self.model, "temperature": 0, "max_tokens": 120, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}}]}]}
        r = self.http.post(self.base + "/chat/completions", json=body)
        r.raise_for_status()
        return _json_from(r.json()["choices"][0]["message"]["content"])

    def transcribe(self, wav: bytes, language: str | None = None) -> str:
        data = {"model": self.model, "response_format": "json"}
        if language:
            data["language"] = language
        r = self.http.post(self.base + "/audio/transcriptions", files={"file": ("speech.wav", wav, "audio/wav")}, data=data)
        r.raise_for_status()
        return str(r.json().get("text") or "").strip()


class AnthropicClient:
    def __init__(self, base_url: str, model: str, api_key: str | None, timeout: float = 30.0):
        if not api_key:
            raise SystemExit("Anthropic provider: set the API key environment variable")
        self.base = (base_url or "https://api.anthropic.com").rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.model = model
        self.http = httpx.Client(timeout=timeout, headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"})

    def _msg(self, system: str, content, max_tokens: int) -> dict:
        r = self.http.post(self.base + "/v1/messages", json={
            "model": self.model, "max_tokens": max_tokens, "temperature": 0, "system": system,
            "messages": [{"role": "user", "content": content}]})
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
        return _json_from(text)

    def chat_json(self, system: str, user: str, schema: dict | None = None, max_tokens: int = 200) -> dict:
        return self._msg(system + "\nReply with the JSON object only.", user, max_tokens)

    def vision_json(self, prompt: str, jpeg_b64: str) -> dict:
        return self._msg("You classify images. Reply with JSON only.", [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64}},
            {"type": "text", "text": prompt}], 120)

    def transcribe(self, wav: bytes, language: str | None = None) -> str:
        raise SystemExit("Anthropic has no speech-to-text endpoint: use stt.backend=local or an OpenAI-compatible STT provider")


def client_for(p) -> OpenAIClient | AnthropicClient:
    """p: config.ProviderConfig"""
    if not p.model:
        raise SystemExit("provider model is not set")
    if p.kind == "anthropic":
        return AnthropicClient(p.base_url, p.model, p.api_key(), p.timeout)
    if p.kind != "openai":
        raise SystemExit(f"unknown provider kind {p.kind!r} (openai | anthropic)")
    return OpenAIClient(p.base_url, p.model, p.api_key(), p.timeout)
