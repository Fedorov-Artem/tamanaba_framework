"""
Environment and other settings for LLM providers.
Uses unified providers.py classes.
"""

import os
from dotenv import load_dotenv
from typing import Any, Dict

# Importing newly refactored classes from our providers module
from providers import OllamaProvider, LlamaProvider, ClaudeProvider, SecureTokenProvider

# Load environment variables (strict separation: secrets only)
load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
SECURE_CLIENT_SECRET = os.getenv("SECURE_CLIENT_SECRET", "")

# Unified token authentication configuration
SCOPE = "ENTERPRISE_API_SCOPE"
SECRET = SECURE_CLIENT_SECRET

# API Gateway Endpoints (urls are configured inside the config file for flexibility)
OLLAMA_URL = "http://localhost:11434/api/chat"
LLAMA_URL = "http://medgemma_llama:8080/v1/chat/completions"
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
SECURE_GATEWAY_URL = "https://gateway.example.com/api/v1/chat/completions"
SECURE_AUTH_URL = "https://gateway.example.com/oauth/token"
SECURE_FILE_STAGING_URL = "https://gateway.example.com/api/v1/files"

# Model Identifier Strings
OLLAMA_MODEL_NAME = "qwen3:14b"
LLAMA_MODEL_NAME = "alibayram/medgemma:27b"
CLAUDE_MODEL_NAME = "claude-3-5-sonnet-latest"
SECURE_GATEWAY_MODEL = "enterprise-model-v1"


# --- Provider Payload Templates ---
# We keep payloads clean. Schema structure injection is handled dynamically by BaseLLM.

payload_ollama: Dict[str, Any] = {
    "model": OLLAMA_MODEL_NAME,
    "temperature": 0.5,
    "think": False,
    "stream": False
}

payload_llama: Dict[str, Any] = {
    "model": LLAMA_MODEL_NAME,
    "temperature": 0.0
}

payload_claude: Dict[str, Any] = {
    "model": CLAUDE_MODEL_NAME,
    "max_tokens": 2048,
    "temperature": 0.0
}

payload_secure_gateway: Dict[str, Any] = {
    "model": SECURE_GATEWAY_MODEL,
    "temperature": 0.0,
    "max_tokens": 2048,
    "stream": False,
}


# --- Standardized Provider Instances ---

llm_ollama = OllamaProvider(
    url=OLLAMA_URL,
    payload_params=payload_ollama,
    num_retries=3,
    timeout=180.0
)

llm_llama_cpp = LlamaProvider(
    url=LLAMA_URL,
    payload_params=payload_llama,
    num_retries=3,
    timeout=360.0
)

llm_claude = ClaudeProvider(
    url=CLAUDE_URL,
    payload_params=payload_claude,
    api_key=CLAUDE_API_KEY,
    num_retries=3,
    timeout=120.0
)

llm_secure_gateway = SecureTokenProvider(
    url=SECURE_GATEWAY_URL,
    payload_params=payload_secure_gateway,
    auth_url=SECURE_AUTH_URL,
    secret=SECRET,
    scope=SCOPE,
    file_staging_url=SECURE_FILE_STAGING_URL,
    num_retries=3,
    timeout=360.0
)

# Active LLM mapping to be imported and used directly in agents
llm_example = llm_claude