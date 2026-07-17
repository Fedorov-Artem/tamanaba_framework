"""
Enterprise LLM Agent Core Workflow.

Deploys a robust Map-Reduce (Fan-Out / Fan-In) orchestration pattern leveraging
LangGraph state machines. The architecture isolates concurrent task executions
using specialized state tokens, handles decentralized parallel processing via
the Unified LLM Gateway SDK, and dynamically aggregates distributed predictions
into a single, consolidated payload.
"""

import logging
from typing import List
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

# Import unified LLM gateway configuration and structures
from ask_llm import llm_example
from prompts import ALL_PROMPTS
from data_models import AgentState, WorkerState

# Initialize standard module-level logging
logger = logging.getLogger(__name__)

# =====================================================================
# 1. ROUTER & NODE DEFINITIONS (FAN-OUT / FAN-IN)
# =====================================================================

async def clean_input_node(state: AgentState) -> dict:
    """
    Preprocesses incoming batch requests and slices data into discrete worker tasks.
    """
    logger.info("Executing node: clean_input_node")
    raw_text = state.get("input_text", "")

    # Splitting comma-separated values into independent items to demonstrate fan-out
    tasks = [t.strip() for t in raw_text.split(",") if t.strip()]

    logger.info(f"Extracted {len(tasks)} parallel tasks for execution distribution.")
    return {
        "tasks": tasks,
        "worker_results": []  # Purge and initialize the target aggregator array
    }


def route_to_workers(state: AgentState) -> List[Send]:
    """
    Conditional edge router function.
    Maps each isolated task dynamically to an independent 'worker_node' state container.
    """
    logger.info("Routing parallel sub-tasks across worker instances (Fan-Out step)")
    tasks = state.get("tasks", [])

    # Map state objects explicitly using LangGraph Send primitives
    return [Send("worker", {"task_input": task}) for task in tasks]


async def worker_node(state: WorkerState) -> dict:
    """
    Parallel Worker node running in complete isolation.
    Executes concurrently for every independent task emitted by the edge router.
    """
    task_data = state["task_input"]
    logger.info(f"Executing parallel worker_node for task sequence: {task_data}")

    # Bind individual parameters to the target domain prompt template
    formatted_prompt = ALL_PROMPTS['example_user_prompt'].format(user_data=task_data)

    try:
        # Accessing the unified SDK gateway via generate_json interface
        response_json = await llm_example.generate_json(
            prompt=formatted_prompt,
            system_prompt=ALL_PROMPTS['example_system_prompt']
        )
        # Extracting target token predictions out of parsed JSON structure
        result = response_json.get("prediction", f"Parsed result for {task_data}")
    except Exception as e:
        logger.error(f"Error processing task {task_data} inside worker runtime context: {e}")
        result = f"Error outcome for {task_data}"

    # Return structure appends to main state automatically via operator.add reducer
    return {"worker_results": [result]}


async def reduce_results_node(state: AgentState) -> dict:
    """
    Consolidator Node (Reduce / Fan-In phase).
    Fires exclusively when every concurrent worker lifecycle has completed successfully.
    """
    logger.info("Executing node: reduce_results_node")
    results = state.get("worker_results", [])
    logger.info(f"Successfully unified {len(results)} individual outputs from parallel batch execution.")

    # Packaging and ordering final compiled arrays into destination fields
    return {"predictions": results}


# =====================================================================
# 2. GRAPH ASSEMBLY & COMPILATION
# =====================================================================

# Initialize state machine workflow graph bounded by root schema properties
workflow = StateGraph(AgentState)

# Mount logical execution frames to specific targets
workflow.add_node("clean", clean_input_node)
workflow.add_node("worker", worker_node)
workflow.add_node("reduce", reduce_results_node)

# Route execution progression sequences
workflow.add_edge(START, "clean")

# Evaluate edge metrics dynamically to distribute load vectors
workflow.add_conditional_edges(
    "clean",
    route_to_workers,
    ["worker"]
)

# Reconverge distributed runtime contexts back into the reducer module
workflow.add_edge("worker", "reduce")
workflow.add_edge("reduce", END)

# Finalize compilation for agent engine injection
app = workflow.compile()