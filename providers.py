"""
Unified Enterprise LLM Gateway SDK.
Uses fail-fast validation with automatic retries for encoding artifacts.
"""

import ast
import asyncio
import base64
import copy
import io
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Type, TypeVar

import httpx
from fastapi import HTTPException
from pydantic import BaseModel
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

T = TypeVar("T", bound=BaseModel)


class SecureTokenManager:
    """
    Thread-safe, async OAuth2 token manager.
    Handles background refreshing using double-checked locking.
    """
    def __init__(self, auth_url: str, client_secret: str, scope: str):
        self._auth_url = auth_url
        self._client_secret = client_secret
        self._scope = scope
        self._lock = asyncio.Lock()

        self._access_token: Optional[str] = None
        self._expires_at: datetime = datetime.now() - timedelta(days=1)
        self._refresh_buffer = timedelta(minutes=5)

    async def get_token(self) -> str:
        """Retrieves a valid token, refreshing it in the background if close to expiry."""
        now = datetime.now()

        # 1. Token is expired or missing: block and refresh
        if not self._access_token or now >= self._expires_at:
            await self._refresh_token()
            return self._access_token

        # 2. Token is close to expiry (within buffer): return current, refresh in background
        if now >= (self._expires_at - self._refresh_buffer):
            token_to_return = self._access_token
            asyncio.create_task(self._refresh_token())
            return token_to_return

        # 3. Token is fresh and safe to use
        return self._access_token

    async def _refresh_token(self):
        """Executes the network request to fetch a new OAuth token."""
        async with self._lock:
            # Double-check: Has another coroutine refreshed it while we waited?
            if self._access_token and datetime.now() < (self._expires_at - self._refresh_buffer):
                return

            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json',
                'X-Request-ID': str(uuid.uuid4()),
                'Authorization': f'Basic {self._client_secret}'
            }
            data = {'scope': self._scope}

            try:
                # SSL verification is disabled to support custom enterprise proxy certificates
                async with httpx.AsyncClient(verify=False, timeout=10) as client:
                    response = await client.post(self._auth_url, headers=headers, data=data)
                    response.raise_for_status()

                    token_data = response.json()
                    self._access_token = token_data.get("access_token")
                    expires_in = token_data.get("expires_in", 1800)
                    self._expires_at = datetime.now() + timedelta(seconds=expires_in)

                    logger.info("OAuth access token successfully refreshed.")

            except Exception as e:
                logger.error(f"Token refresh sequence failed: {e}")
                if not self._access_token:
                    raise HTTPException(status_code=503, detail="Upstream Authentication Failed")


class BaseLLM(ABC):
    """
    Abstract Base Class for all LLM providers.
    Enforces standardized retry logic, fail-fast validation, and JSON parsing.
    """
    def __init__(self, url: str, payload_params: Dict[str, Any], num_retries: int = 2, timeout: float = 120.0):
        self.url = url
        self.payload_params = payload_params
        self.num_retries = num_retries
        self.timeout = timeout

    async def _retry_with_backoff(self, request_func, *args, **kwargs):
        """
        Execute request_func with retry logic and exponential backoff.
        """
        if self.num_retries <= 1:
            return await request_func(*args, **kwargs)

        last_exception = None
        for attempt in range(self.num_retries):
            try:
                return await request_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.num_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.num_retries}). "
                        f"Retrying in {wait_time} seconds... Error: {e}"
                    )
                    await asyncio.sleep(wait_time)

        raise last_exception

    @abstractmethod
    async def _raw_ask(self, prompt: str, system_prompt: str,
                       format_schema: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Low-level communication with the specific model API. Must be implemented by subclasses.
        """
        pass

    async def generate_json(self, prompt: str, system_prompt: str,
                            schema_class: Optional[Type[BaseModel]] = None, **kwargs) -> Dict[str, Any]:
        """
        The main public unified entrypoint.
        Generates, validates, extracts structured JSON schemas, and logs traces to Phoenix.
        """

        async def _generate_and_extract():
            format_schema = schema_class.model_json_schema() if schema_class else None

            # --- START ARIZE PHOENIX TRACING ---
            # Создаем спан и указываем, что это вызов LLM
            with tracer.start_as_current_span(
                    "llm_generate",
                    attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM"}
            ) as span:

                # 1. Записываем входные данные в трейс
                # Пытаемся вытащить имя модели из параметров, если оно там есть
                model_name = self.payload_params.get("model", "unknown")
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
                span.set_attribute(SpanAttributes.LLM_SYSTEM_PROMPT, system_prompt)
                span.set_attribute(SpanAttributes.LLM_INPUT_MESSAGES, prompt)

                # Сохраняем все параметры payload (temperature, max_tokens и т.д.)
                span.set_attribute(
                    SpanAttributes.LLM_INVOCATION_PARAMETERS,
                    json.dumps(self.payload_params)
                )

                try:
                    # 2. Фактический вызов API
                    raw_text = await self._raw_ask(prompt, system_prompt, format_schema=format_schema, **kwargs)

                    # 3. Записываем сырой ответ модели в трейс до любых проверок
                    span.set_attribute(SpanAttributes.LLM_OUTPUT_MESSAGES, raw_text)

                    # Fail-fast проверка на кракозябры
                    if self._contains_unicode_artifacts(raw_text):
                        raise ValueError("Detected encoding corruption (Unicode artifacts) in LLM response.")

                    # Извлечение JSON
                    return self._extract_json(raw_text)

                except Exception as e:
                    # Если произошла ошибка (сеть, невалидный JSON, иероглифы) - Phoenix подсветит трейс красным
                    span.record_exception(e)
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    raise
            # --- END TRACING ---

        return await self._retry_with_backoff(_generate_and_extract)

    def _contains_unicode_artifacts(self, text: str) -> bool:
        """Detects CJK/hieroglyphs encoding glitches in localized outputs."""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Locates, extracts, and parses the JSON structure from the response."""
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError("Model output did not contain a valid JSON object")

        try:
            fixed_text = match.group(0)
            fixed_text = re.sub(r"```json\s?|```", "", fixed_text).strip()

            try:
                # ast.literal_eval handles Python-like key/value syntax slip-ups beautifully
                return ast.literal_eval(fixed_text)
            except Exception:
                try:
                    return json.loads(fixed_text)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Strict JSON parsing failed: {e}")
        except Exception as e:
            raise ValueError(f"Error while processing structured output: {e}")


# --- Provider Implementations ---

class OllamaProvider(BaseLLM):
    """
    Provider for local Ollama deployments.
    """
    async def _raw_ask(self, prompt: str, system_prompt: str,
                       format_schema: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        payload = self.payload_params.copy()
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        if format_schema:
            payload["format"] = format_schema

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, json=payload)
            if response.is_error:
                try:
                    err = response.json()
                except Exception:
                    err = response.text
                logger.error(f"Ollama Error ({response.status_code}): {err}. Payload: {payload}")
                response.raise_for_status()

            return response.json()["message"]["content"]


class LlamaProvider(BaseLLM):
    """
    Provider optimized for llama.cpp, vLLM, or other self-hosted OpenAI-compatible JSON-mode engines.
    """
    async def _raw_ask(self, prompt: str, system_prompt: str,
                       format_schema: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        response_format: Dict[str, Any] = {"type": "json_object"}
        if format_schema:
            response_format["schema"] = format_schema

        payload = self.payload_params.copy()
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, json=payload)
            if response.is_error:
                try:
                    err = response.json()
                except Exception:
                    err = response.text
                logger.error(f"Llama/vLLM Error ({response.status_code}): {err}. Payload: {payload}")
                response.raise_for_status()

            return response.json()['choices'][0]['message']['content']


class StructuredLLMProvider(BaseLLM):
    """
    Provider for API endpoints enforcing strict OpenAI-like 'json_schema' formatting.
    """
    def __init__(self, url: str, payload_params: Dict[str, Any], headers: Dict[str, Any], num_retries: int = 2, timeout: float = 120):
        super().__init__(url, payload_params, num_retries, timeout)
        self.headers = headers

    async def _raw_ask(self, prompt: str, system_prompt: str,
                       format_schema: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        payload = self.payload_params.copy()

        if format_schema:
            payload['response_format'] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "StructuredOutputResponse",
                    "strict": True,
                    "schema": format_schema
                }
            }

        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=self.headers, json=payload)
            if response.is_error:
                try:
                    err = response.json()
                except Exception:
                    err = response.text
                logger.error(f"Structured API Error ({response.status_code}): {err}. Payload: {payload}")
                response.raise_for_status()

            return response.json()['choices'][0]['message']['content']


class SecureTokenProvider(BaseLLM):
    """
    Handles OAuth token-rotated custom APIs (e.g., corporate enterprise gateways).
    Manages background authorization, optional multi-format document staging, and execution.
    """
    def __init__(self, url: str, payload_params: Dict[str, Any], auth_url: str, secret: str, scope: str,
                 file_staging_url: Optional[str] = None, num_retries: int = 2, timeout: int = 120):
        super().__init__(url, payload_params, num_retries, timeout)
        self.file_staging_url = file_staging_url
        self.token_manager = SecureTokenManager(auth_url, secret, scope)

    async def _delete_staged_file(self, file_id: str, access_token: str):
        if not self.file_staging_url:
            return

        delete_url = f"{self.file_staging_url}/{file_id}/delete"
        headers = {"Authorization": f"Bearer {access_token}"}

        async with httpx.AsyncClient(verify=False) as client:
            try:
                response = await client.post(delete_url, headers=headers)
                response.raise_for_status()
            except Exception as e:
                logger.warning(f"Failed to clean up file {file_id}: {e}")

    async def _upload_staged_file(self, file_bytes: bytes, access_token: str, mime_type: str = "text/plain") -> Optional[str]:
        if not self.file_staging_url:
            return None

        headers = {"Authorization": f"Bearer {access_token}"}
        mime_map = {
            'text/plain': '.txt',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/pdf': '.pdf',
        }
        filename = f"staged_document{mime_map.get(mime_type.lower(), '.bin')}"

        file_buffer = io.BytesIO(file_bytes)
        files = {'file': (filename, file_buffer, mime_type)}
        data = {'purpose': 'general'}

        async with httpx.AsyncClient(timeout=60, verify=False) as client:
            try:
                response = await client.post(self.file_staging_url, headers=headers, data=data, files=files)
                response.raise_for_status()
                return response.json().get('id')
            except Exception as e:
                logger.error(f"Staged file upload failed: {e}")
                return None
            finally:
                file_buffer.close()

    async def _raw_ask(self, prompt: str, system_prompt: str,
                       format_schema: Optional[Dict[str, Any]] = None,
                       att_file: Optional[str] = None,
                       att_file_type: Optional[str] = None, **kwargs) -> str:
        access_token = await self.token_manager.get_token()

        payload = self.payload_params.copy()
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        if format_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "schema": format_schema
            }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        file_id: Optional[str] = None

        if att_file:
            try:
                file_bytes = base64.b64decode(att_file)
                file_id = await self._upload_staged_file(file_bytes, access_token, att_file_type or "text/plain")
            except Exception as e:
                logger.error(f"Attachment decode failed: {e}. Executing raw query without attachment.")

        if file_id:
            payload["messages"][1]["attachments"] = [file_id]
            payload["function_call"] = "auto"

        try:
            async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                response = await client.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
        finally:
            if file_id:
                await self._delete_staged_file(file_id, access_token)


class ClaudeProvider(BaseLLM):
    """
    Provider designed for Anthropic Claude.
    Inherits from BaseLLM and uses tool-calling mechanisms to enforce strict JSON schemas.
    """
    def __init__(self, url: str, payload_params: Dict[str, Any], api_key: str,
                 api_version: str = "2023-06-01", num_retries: int = 3, timeout: float = 120.0):
        super().__init__(url, payload_params, num_retries, timeout)
        self.api_key = api_key
        self.api_version = api_version
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    async def _raw_ask(self, prompt: str, system_prompt: str,
                       format_schema: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        payload = copy.deepcopy(self.payload_params)

        if system_prompt:
            payload["system"] = system_prompt

        # Support for Anthropic prompt caching (handling list elements in content)
        first_msg = payload.get("messages", [{}])[0]
        if isinstance(first_msg.get("content"), list):
            first_msg["content"].append({
                "type": "text",
                "text": f"\n{prompt}"
            })
        else:
            payload["messages"] = [
                {"role": "user", "content": prompt}
            ]

        # Enforce strict structured outputs by converting the schema into an Anthropic-compatible tool call
        if format_schema:
            payload["tools"] = [{
                "name": "structured_output_response",
                "description": "Output schema instructions.",
                "input_schema": format_schema
            }]
            payload["tool_choice"] = {"type": "tool", "name": "structured_output_response"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=self.headers, json=payload)

            # Let _retry_with_backoff handle rate limits (429) or network hiccups
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                sleep_time = int(retry_after) if retry_after else 10
                logger.warning(f"Rate limited (429). Retry scheduled after sleeping {sleep_time} seconds.")
                await asyncio.sleep(sleep_time)
                response.raise_for_status()

            response.raise_for_status()
            result = response.json()
            content_blocks = result.get("content", [])

            if format_schema:
                tool_use_block = next((b for b in content_blocks if b.get("type") == "tool_use"), None)
                if not tool_use_block:
                    logger.error("Claude returned plain text instead of structured tool call.")
                    raise ValueError("Model failed to adhere to the requested schema.")

                # Serializing inputs to string so BaseLLM._extract_json can clean/validate it
                return json.dumps(tool_use_block.get("input"))
            else:
                text_block = next((b for b in content_blocks if b.get("type") == "text"), None)
                return text_block.get("text", "") if text_block else ""

