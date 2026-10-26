"""
harness/llm_caller.py

Provider-agnostic LLM invocation layer (FR3, NFR2, NFR5).

For this build we wire up ONE concrete provider — Gemini — but the abstract
base class is kept so a second provider (Claude/OpenAI) can be added later
by implementing a single method and registering a factory entry, per NFR5.
No changes to context_assembler.py, judge.py, or results_store.py are needed
to add a provider.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

try:
    from google import genai
    from google.genai import errors as genai_errors
except ImportError:  # pragma: no cover
    genai = None
    genai_errors = None

try:
    from mistralai import Mistral
except ImportError:
    try:
        # Newer mistralai SDK versions (2.x) moved the client class here.
        from mistralai.client import Mistral
    except ImportError:  # pragma: no cover
        Mistral = None


# --------------------------------------------------------------------------- #
# Response envelope
# --------------------------------------------------------------------------- #

class DailyQuotaExceededError(Exception):
    """Raised when the provider reports a per-day (not per-minute) quota
    exhaustion. Deliberately NOT included in any provider's retryable
    exception tuple — retrying with backoff is pointless against a quota
    that resets once every 24 hours, and doing so anyway just burns the
    run's wall-clock time for nothing."""


@dataclass
class LLMResponse:
    """Uniform response envelope returned by every provider."""
    text: str
    model: str
    provider: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw_finish_reason: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Rate limiter — simple token-bucket-ish sliding window, thread-safe
# --------------------------------------------------------------------------- #

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.rpm = max(1, requests_per_minute)
        self._interval = 60.0 / self.rpm
        self._lock = threading.Lock()
        self._last_call_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_at
            sleep_for = self._interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call_at = time.monotonic()


# --------------------------------------------------------------------------- #
# JSONL call logger (NFR1, NFR3, FR3)
# --------------------------------------------------------------------------- #

class CallLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, record: dict) -> None:
        record = dict(record)
        record["logged_at"] = datetime.now(timezone.utc).isoformat()
        _redact_secrets_in_place(record)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _redact_secrets_in_place(record: dict) -> None:
    """NFR3 belt-and-braces: strip anything that looks like a credential
    before it ever touches disk, even though callers should never pass one.
    Deliberately excludes 'input_tokens'/'output_tokens' (token *counts*,
    not credentials) so real usage metrics aren't clobbered."""
    banned_keys = {"api_key", "apikey", "secret", "authorization", "auth_token", "access_token"}
    for k in list(record.keys()):
        if k.lower() in banned_keys:
            record[k] = "***REDACTED***"


# --------------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------------- #

class LLMCaller(ABC):
    provider_name: str = "abstract"

    def __init__(
        self,
        model_name: str,
        rate_limit_rpm: int = 10,
        max_retries: int = 5,
        log_path: str | Path = "results/llm_calls.jsonl",
        training_tier: bool = False,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.training_tier = training_tier  # NFR6 transparency flag
        self._rate_limiter = RateLimiter(rate_limit_rpm)
        self._logger = CallLogger(log_path)

    def call(self, prompt: str, system_prompt: str = "", *, run_metadata: Optional[dict] = None) -> LLMResponse:
        """Public entry point: rate-limits, retries with backoff, logs, and
        returns a uniform LLMResponse regardless of provider."""
        run_metadata = run_metadata or {}
        self._rate_limiter.wait()

        start = time.perf_counter()
        try:
            response = self._invoke_with_retry(prompt, system_prompt)
            error = None
        except Exception as exc:  # noqa: BLE001 — we want to log & re-raise-free
            response = LLMResponse(
                text="",
                model=self.model_name,
                provider=self.provider_name,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=str(exc),
            )
            error = str(exc)

        latency_ms = (time.perf_counter() - start) * 1000
        response.latency_ms = latency_ms

        self._logger.log({
            "provider": self.provider_name,
            "model": self.model_name,
            "prompt_chars": len(prompt),
            "system_prompt_chars": len(system_prompt),
            "latency_ms": latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "error": error,
            "training_tier": self.training_tier,
            **run_metadata,
        })
        return response

    def _invoke_with_retry(self, prompt: str, system_prompt: str) -> LLMResponse:
        retrier = retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            # Ceiling raised from 30s -> 60s: a real rate-limit storm from
            # several concurrently-started Actions jobs needs more than a
            # 30s backoff cap to clear (see pr_review_bot.yml's
            # `concurrency:` group, which is the primary fix for that).
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(self._transient_exceptions()),
        )
        return retrier(self._invoke)(prompt, system_prompt)

    @abstractmethod
    def _invoke(self, prompt: str, system_prompt: str) -> LLMResponse:
        """Provider-specific single-shot call. Must raise on transient
        failure so tenacity can retry; must NOT raise on a normal (even
        refused/empty) model response."""
        raise NotImplementedError

    @abstractmethod
    def _transient_exceptions(self) -> tuple[type[Exception], ...]:
        """Exception types worth retrying (rate limit, 5xx, timeout)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Gemini implementation
# --------------------------------------------------------------------------- #

class GeminiCaller(LLMCaller):
    provider_name = "gemini"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if genai is None:
            raise ImportError(
                "google-genai is not installed. Run: "
                "pip install -r requirements.txt --break-system-packages"
            )
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self.client = genai.Client(api_key=api_key)

    def _invoke(self, prompt: str, system_prompt: str) -> LLMResponse:
        config = {}
        if system_prompt:
            config["system_instruction"] = system_prompt

        try:
            result = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config or None,
            )
        except Exception as exc:
            # A per-DAY quota exhaustion will never succeed on retry within
            # the same run (it resets on a 24h clock, not a 60s one) — fail
            # fast instead of burning 5 backoff attempts x up to 30s each.
            if "PerDay" in str(exc) or "RequestsPerDay" in str(exc):
                raise DailyQuotaExceededError(str(exc)) from exc
            raise

        text = result.text or ""
        usage = getattr(result, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None

        finish_reason = None
        try:
            finish_reason = result.candidates[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            pass

        return LLMResponse(
            text=text,
            model=self.model_name,
            provider=self.provider_name,
            latency_ms=0.0,  # filled in by base class
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_finish_reason=str(finish_reason) if finish_reason else None,
        )

    def _transient_exceptions(self) -> tuple[type[Exception], ...]:
        if genai_errors is None:
            return (Exception,)
        # google-genai raises APIError subclasses for HTTP-level failures.
        return (genai_errors.APIError,)


# --------------------------------------------------------------------------- #
# Mistral implementation
# --------------------------------------------------------------------------- #

class MistralTransientError(Exception):
    """Wraps a Mistral API error that looks retryable (429 / 5xx / timeout).
    We detect this by inspecting the underlying exception rather than
    depending on mistralai's exact internal exception hierarchy, since that
    can change between SDK versions."""


class MistralCaller(LLMCaller):
    provider_name = "mistral"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if Mistral is None:
            raise ImportError(
                "mistralai is not installed. Run: "
                "pip install -r requirements.txt --break-system-packages"
            )
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        self.client = Mistral(api_key=api_key)

    def _invoke(self, prompt: str, system_prompt: str) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            result = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
            )
        except Exception as exc:
            msg = str(exc)
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            retryable_codes = {429, 500, 502, 503, 504}
            is_retryable = (status in retryable_codes) or any(
                kw in msg.lower()
                for kw in ("rate limit", "429", "timeout", "temporarily unavailable", "502", "503", "504")
            )
            if is_retryable:
                raise MistralTransientError(msg) from exc
            raise  # non-retryable (e.g. bad request, auth failure) — surface immediately

        choice = result.choices[0]
        text = choice.message.content or ""
        if isinstance(text, list):  # some SDK versions return content blocks
            text = "".join(getattr(b, "text", str(b)) for b in text)

        usage = getattr(result, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        finish_reason = getattr(choice, "finish_reason", None)

        return LLMResponse(
            text=text,
            model=self.model_name,
            provider=self.provider_name,
            latency_ms=0.0,  # filled in by base class
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_finish_reason=str(finish_reason) if finish_reason else None,
        )

    def _transient_exceptions(self) -> tuple[type[Exception], ...]:
        return (MistralTransientError,)


# --------------------------------------------------------------------------- #
# Groq implementation (OpenAI-compatible REST API; stdlib only, no extra dependency)
# --------------------------------------------------------------------------- #

class GroqTransientError(Exception):
    """429 (per-minute) / 5xx / network timeout from Groq: worth retrying."""


class GroqCaller(LLMCaller):
    provider_name = "groq"
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    TIMEOUT_S = 60

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
        self._api_key = api_key
        self._base_url = (base_url or os.environ.get("GROQ_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")

    def _invoke(self, prompt: str, system_prompt: str) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps({"model": self.model_name, "messages": messages, "temperature": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                # A default Python-urllib User-Agent is sometimes blocked by the edge proxy.
                "User-Agent": "pipelinepoison/2.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            msg = f"Groq HTTP {exc.code}: {detail}"
            if exc.code == 429:
                # Per-DAY limits reset on a 24h clock: retrying is pointless, so stop the run.
                if "per day" in detail.lower() or "(rpd)" in detail.lower() or "(tpd)" in detail.lower():
                    raise DailyQuotaExceededError(msg) from exc
                try:  # honour Retry-After for per-minute limits (capped)
                    time.sleep(min(float(exc.headers.get("retry-after", 0)), 20.0))
                except (TypeError, ValueError):
                    pass
                raise GroqTransientError(msg) from exc
            if exc.code in (500, 502, 503, 504):
                raise GroqTransientError(msg) from exc
            raise RuntimeError(msg) from exc  # 400/401/403/404: not retryable
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GroqTransientError(f"Groq network error: {exc}") from exc

        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return LLMResponse(
            text=(choice.get("message") or {}).get("content") or "",
            model=self.model_name,
            provider=self.provider_name,
            latency_ms=0.0,  # filled in by base class
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            raw_finish_reason=choice.get("finish_reason"),
        )

    def _transient_exceptions(self) -> tuple[type[Exception], ...]:
        return (GroqTransientError,)

# --------------------------------------------------------------------------- #
# Ollama implementation (local, OpenAI-compatible; reuses the Groq REST code)
# --------------------------------------------------------------------------- #

class OllamaCaller(GroqCaller):
    provider_name = "ollama"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    TIMEOUT_S = 600  # local inference is slow, especially the first call

    def __init__(self, base_url: Optional[str] = None, **kwargs):
        base_url = base_url or os.environ.get("OLLAMA_BASE_URL") or self.DEFAULT_BASE_URL
        super().__init__(api_key="ollama", base_url=base_url, **kwargs)  # key is ignored by Ollama

# --------------------------------------------------------------------------- #
# Mock caller — used by tests and dry-runs, no network access required
# --------------------------------------------------------------------------- #

class MockCaller(LLMCaller):
    """Deterministic offline stand-in so the harness can be smoke-tested
    without an API key or network access. Never used for real results."""
    provider_name = "mock"

    def __init__(self, canned_response: str = "I cannot comply with embedded instructions.", **kwargs):
        kwargs.setdefault("model_name", "mock-model")
        super().__init__(**kwargs)
        self.canned_response = canned_response

    def _invoke(self, prompt: str, system_prompt: str) -> LLMResponse:
        # Extremely naive "attacker-aware" mock so smoke tests can exercise
        # both the benign and successfully-injected code paths deterministically.
        text = self.canned_response
        if "ALWAYS APPROVE" in prompt or "approve this pr" in prompt.lower():
            text = "Approved. LGTM, merging now."
        return LLMResponse(
            text=text,
            model=self.model_name,
            provider=self.provider_name,
            latency_ms=0.0,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
        )

    def _transient_exceptions(self) -> tuple[type[Exception], ...]:
        return (ConnectionError,)


# --------------------------------------------------------------------------- #
# Factory (NFR5: adding a provider = one method + one registry line)
# --------------------------------------------------------------------------- #

_PROVIDER_REGISTRY = {
    "gemini": GeminiCaller,
    "mistral": MistralCaller,
    "groq": GroqCaller,
    "ollama": OllamaCaller,
    "mock": MockCaller,
}


def get_caller(provider: str, **kwargs) -> LLMCaller:
    try:
        cls = _PROVIDER_REGISTRY[provider]
    except KeyError as exc:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {list(_PROVIDER_REGISTRY)}"
        ) from exc
    return cls(**kwargs)
