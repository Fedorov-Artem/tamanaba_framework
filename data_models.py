"""
Tamanaba Framework State and Data Schemas.

Defines the unified Pydantic schemas for HTTP request/response validation
and the TypedDict state containers for LangGraph workflow orchestration.
"""

from typing import List, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
import operator


# =====================================================================
# 1. HTTP API VALIDATION SCHEMAS (PYDANTIC)
# =====================================================================

class EvaluationGroundTruth(BaseModel):
    """Holds expected outcomes for online telemetry evaluation metrics."""
    expected_predictions: List[str] = Field(default_factory=list)


class DataEntry(BaseModel):
    """Represents a single payload item within an incoming batch request."""
    input_text: str
    ground_truth: Optional[EvaluationGroundTruth] = None


class AnalysisResponse(BaseModel):
    """Standardized structured exit schema for the API server gateway."""
    predictions: List[str]


# =====================================================================
# 2. LANGGRAPH WORKFLOW STATE SCHEMAS (TYPEDDICT)
# =====================================================================

class WorkerState(TypedDict):
    """Isolated, thread-safe state context passed to each parallel map instance."""
    task_input: str


class AgentState(TypedDict):
    """Main orchestrator state machine schema holding execution history."""
    input_text: str
    tasks: List[str]  # Sub-tasks distributed across workers (Fan-Out phase)

    # Annotated with operator.add to allow concurrent worker lists
    # to automatically append and merge upon completion (Fan-In phase)
    worker_results: Annotated[List[str], operator.add]
    predictions: List[str]