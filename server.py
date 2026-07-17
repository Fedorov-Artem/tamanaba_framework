"""
Tamanaba Framework FastAPI Server.

Provides a production-ready HTTP gateway for orchestrating LangGraph agents.
Handles automatic OpenTelemetry tracking via Arize Phoenix, fail-safe 
Pydantic validation, truncation of heavy logging payloads,
and real-time evaluation annotation submission.
"""

import logging
import sys
import os
import json

from typing import Dict
from pydantic import BaseModel, Field

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from data_models import AnalysisResponse, DataEntry

# =====================================================================
# 1. TELEMETRY & TRACING CONFIGURATION (MUST RUN BEFORE IMPORTING AGENT)
# =====================================================================

from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from phoenix.client import Client

# Standard docker-friendly logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize Phoenix tracer registration using endpoints from environment variables
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
register(
    endpoint=PHOENIX_ENDPOINT,
    protocol="grpc"
)

PHOENIX_ENDPOINT_ANNOTATION = os.getenv("PHOENIX_UI_ENDPOINT")
phoenix_client = Client(base_url=PHOENIX_ENDPOINT_ANNOTATION)

# Instrument the active tracing context
LangChainInstrumentor().instrument()
tracer = trace.get_tracer(__name__)

# =====================================================================
# 2. AGENT IMPORT
# =====================================================================

from agent import app as agent_app


# =====================================================================
# 3. FASTAPI SETUP & EXCEPTION HANDLERS
# =====================================================================

app = FastAPI(title="Tamanaba AI Agent Server")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        # 1. Extract granular validation failure details from Pydantic
        errors = exc.errors()

        # 2. Safely format and truncate request body to protect logs from memory bloat
        raw_body = exc.body
        truncated_body = "None"

        if raw_body is not None:
            try:
                # Convert body payload into a compact JSON string structure
                body_str = json.dumps(raw_body, ensure_ascii=False)
            except Exception:
                # Fallback to standard string representation if payload bytes break json.dumps
                body_str = str(raw_body)

            # Restrict log length to avoid flooding stdout buffers
            limit = 500
            if len(body_str) > limit:
                truncated_body = body_str[:limit] + f"... [TRUNCATED! Total characters: {len(body_str)}]"
            else:
                truncated_body = body_str

        # 3. Write parsed, compact, and sanitized  information to logs
        logger.error(
            f"HTTP 422 Validation Error on {request.method} {request.url.path}\n"
            f"Errors: {errors}\n"
            f"Received Body (Truncated): {truncated_body}"
        )

        # 4. Prevent unexpected 500 serialization crashes during response generation
        # Passing errors through jsonable_encoder guarantees safe JSON serialization.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": jsonable_encoder(errors)},
        )

    except Exception as handler_exc:
        # Critical guard clause to capture errors occurring inside the handler itself
        logger.critical(
            f"Critical failure inside validation_exception_handler lifecycle: {handler_exc}",
            exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal Server Error during validation"},
        )


# =====================================================================
# 4. ONLINE EVALUATIONS (PHOENIX ANNOTATIONS)
# =====================================================================

async def evaluate_outputs(predictions: list, ground_truth: list, target_span_id: str):
    """Compares agent output to expectations and submits scores to Phoenix."""
    total = len(ground_truth)
    if total == 0:
        return

    correct = sum(1 for p in predictions if p in ground_truth)
    score = round((correct / total) * 100)

    explanation = f"Predicted: {predictions}\nExpected: {ground_truth}"
    logger.info(f"Trace evaluation score for span {target_span_id}: {score}%")

    try:
        phoenix_client.spans.add_span_annotation(
            span_id=target_span_id,
            annotation_name="Prediction Match Score",
            annotator_kind="CODE",  # Required for Phoenix v14 architecture compatibility
            label=f"{score}%",
            score=score,
            explanation=explanation
        )
    except Exception as e:
        logger.error(f"Failed to submit trace evaluation to Phoenix: {e}")


# =====================================================================
# 5. ENDPOINTS
# =====================================================================

@app.post("/example_endpoint", response_model=AnalysisResponse)
async def example_endpoint(request: Dict[str, DataEntry]):
    """Orchestrates agent execution under an active telemetry trace."""
    with tracer.start_as_current_span("agent_execution") as span:
        span_context = span.get_span_context()
        trace_id_str = format(span_context.trace_id, "032x")
        span_id_str = format(span_context.span_id, "016x")

        final_predictions = []

        for record_id, entry in request.items():
            # Invoke the generic agent workflow asynchronously
            # input_text property matches clean_input_node tracking expectations
            result = await agent_app.ainvoke({
                "input_text": entry.input_text,
                "processed_data": "",
                "predictions": []
            })

            final_predictions = result.get("predictions", [])
            logger.info(f"Execution finished for record {record_id}. Outputs: {final_predictions}")

            # Run evaluation if ground-truth expectation metadata was included in runtime payload
            if entry.ground_truth and entry.ground_truth.expected_predictions:
                # OTel ID Fallback logic
                if not span_id_str and trace_id_str:
                    span_id_str = trace_id_str[:16]

                if span_id_str:
                    await evaluate_outputs(
                        predictions=final_predictions,
                        ground_truth=entry.ground_truth.expected_predictions,
                        target_span_id=span_id_str
                    )
                else:
                    logger.error("Unable to execute live scoring: OTel span ID context unavailable.")

        return AnalysisResponse(predictions=final_predictions)