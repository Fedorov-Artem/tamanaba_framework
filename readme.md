# Tamanaba Template
### *Production-Ready AI Agent Orchestration & Observability*

Tamanaba is an enterprise-grade template designed for building, deploying, and monitoring scalable AI agents. It bridges the gap between experimental LLM scripts and highly available production microservices by combining graph-based execution, strict data validation, resilient LLM interactions, and first-class telemetry.

---

## Core Architecture & Stack

*   **Orchestration Layer (`LangGraph`):** Implements advanced parallel processing patterns (Fan-Out / Fan-In). Tasks are distributed across concurrent workers and automatically reduced/accumulated into a unified state using `TypedDict` and `operator.add`.
*   **API Gateway (`FastAPI`):** A robust HTTP entry point that handles batch request processing. It features fail-safe `Pydantic` validation diagnostics, payload truncation to prevent memory bloat in logs, and graceful error handling.
*   **Strict Data Contracts (`Pydantic`):** Unified schemas enforce strict I/O validation (`DataEntry`, `AnalysisResponse`, `EvaluationGroundTruth`), ensuring the graph only receives clean, typed data.

---

## Resilient LLM Engine

At the heart of the template is a custom, highly reliable abstraction layer (`BaseLLM` / `ask_llm.py`) built on `httpx` for interacting with various LLM providers (OpenAI, local Llama.cpp, etc.):

*   **Exponential Backoff Retries:** Automatically recovers from network hiccups or API rate limits.
*   **Fail-Fast Validation:** Proactively detects text generation glitches (e.g., Unicode/CJK artifacts in localized outputs) and immediately triggers a retry.
*   **Bulletproof JSON Extraction:** Uses aggressive Regex mapping combined with `ast.literal_eval` and standard `json.loads` as fallbacks to successfully parse structured outputs even when the model hallucinates formatting (like missing quotes or trailing commas).

---

## First-Class Observability (Arize Phoenix)

The template includes native OpenTelemetry (OTel) integration seamlessly injected into both the API layer and the LLM Base class:

*   **Deep LLM Tracing:** Captures exact prompts, system instructions, invocation parameters (temperature, max tokens), and raw responses directly into Arize Phoenix dashboards.
*   **Online Evaluations:** Automatically compares agent outputs against ground-truth datasets and submits execution metrics (Prediction Match Scores) as trace annotations via the Phoenix Client.
*   **Isolated Infrastructure:** Fully containerized via `Docker Compose` within an isolated `ai-network`, utilizing dynamic environment variables to prevent internal DNS collisions and port conflicts between multiple parallel deployments.