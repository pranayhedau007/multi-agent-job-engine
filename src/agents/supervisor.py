"""
Author: Pranay Hedau
Purpose: Supervisor Agent routes tasks to specialist agents, manages approval gates."""

import logging
from src.models.schemas import AgentAction

logger = logging.getLogger(__name__)

# Which agents run for each action type
ROUTING = {
    AgentAction.SEARCH_JOBS: ["job_scout"],
    AgentAction.ANALYZE_FIT: ["job_scout", "fit_analyst"],
    AgentAction.TAILOR_RESUME: ["job_scout", "fit_analyst", "resume_tailor"],
    AgentAction.DRAFT_OUTREACH: ["job_scout", "fit_analyst", "resume_tailor", "outreach"],
    AgentAction.FULL_PIPELINE: ["job_scout", "fit_analyst", "resume_tailor", "outreach"],
}

# Agents that require human approval before the next agent runs
APPROVAL_GATES = {"fit_analyst", "resume_tailor"}


def get_next_agent(current: str, action: AgentAction) -> str | None:
    """Purpose: Determine next agent in the pipeline. Returns None if done."""
    seq = ROUTING.get(action, [])
    if not current or current not in seq:
        return seq[0] if seq else None
    idx = seq.index(current)
    return seq[idx + 1] if idx + 1 < len(seq) else None


def should_request_approval(current: str, action: AgentAction) -> bool:
    """Purpose: Check if human approval is needed before proceeding.
    if step requries approval or more steps are remaining, it returns true else false
    """
    seq = ROUTING.get(action, [])
    if current in APPROVAL_GATES:
        idx = seq.index(current) if current in seq else -1
        return idx + 1 < len(seq) #it checks if after the current action still more steps are remaining then return true
    return False


def route_decision(current: str, action: AgentAction, approved: bool) -> str:
    """Purpose: Main routing logic for LangGraph conditional edges.
    
    Returns: "continue", "wait_for_human", or "end"
    """
    if should_request_approval(current, action) and not approved:
        logger.info(f"Supervisor: pausing after {current} for approval")
        return "wait_for_human"
    if get_next_agent(current, action) is None:
        logger.info("Supervisor: workflow complete")
        return "end"
    logger.info(f"Supervisor: {current} -> {get_next_agent(current, action)}")
    return "continue"