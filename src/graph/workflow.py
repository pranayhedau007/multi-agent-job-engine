"""
Author: Pranay Hedau
Purpose: LangGraph workflow - the full agent pipeline as a state machine.
Which agent will be called after the current agent, deatiled explanation in the architectural diag in readme 
"""

import logging
from langgraph.graph import StateGraph, START, END

from src.graph.state import AgentState
from src.graph.nodes import (
    job_scout_node, fit_analyst_node,
    resume_tailor_node, outreach_node,
    human_approval_node,
)
from src.agents.supervisor import route_decision
from src.models.schemas import AgentAction, UserPreferences

logger = logging.getLogger(__name__)


def _after_scout(state: AgentState) -> str:
    if state.get("requested_action") == AgentAction.SEARCH_JOBS:
        return END
    return "fit_analyst"


def _after_fit(state: AgentState) -> str:
    action = state.get("requested_action", AgentAction.FULL_PIPELINE)
    d = route_decision("fit_analyst", action, state.get("human_approved", False))
    if d == "wait_for_human":
        return "human_review_fit"
    return END if d == "end" else "resume_tailor"


def _after_review_fit(state: AgentState) -> str:
    return "resume_tailor" if state.get("human_approved") else END


def _after_tailor(state: AgentState) -> str:
    action = state.get("requested_action", AgentAction.FULL_PIPELINE)
    d = route_decision("resume_tailor", action, state.get("human_approved", False))
    if d == "wait_for_human":
        return "human_review_tailor"
    return END if d == "end" else "outreach"


def _after_review_tailor(state: AgentState) -> str:
    return "outreach" if state.get("human_approved") else END


def build_workflow() -> StateGraph:
    """Purpose: Build the LangGraph state machine.
    
    Graph: START -> job_scout -> fit_analyst -> [approval] -> resume_tailor -> [approval] -> outreach -> END
    """
    g = StateGraph(AgentState)

    g.add_node("job_scout", job_scout_node)
    g.add_node("fit_analyst", fit_analyst_node)
    g.add_node("human_review_fit", human_approval_node)
    g.add_node("resume_tailor", resume_tailor_node)
    g.add_node("human_review_tailor", human_approval_node)
    g.add_node("outreach", outreach_node)

    g.add_edge(START, "job_scout")
    g.add_conditional_edges("job_scout", _after_scout)
    g.add_conditional_edges("fit_analyst", _after_fit)
    g.add_conditional_edges("human_review_fit", _after_review_fit)
    g.add_conditional_edges("resume_tailor", _after_tailor)
    g.add_conditional_edges("human_review_tailor", _after_review_tailor)
    g.add_edge("outreach", END)

    return g


def compile_workflow():
    return build_workflow().compile()


def run_pipeline(
    resume_text: str,
    target_roles: list[str],
    locations: list[str] | None = None,
    tech_stack: list[str] | None = None,
    visa_required: bool = True,
    action: AgentAction = AgentAction.FULL_PIPELINE,
    auto_approve: bool = False,
) -> AgentState:
    """Run the full agent pipeline.
    
    Args:
        resume_text: Parsed resume content
        target_roles: Roles to search for
        locations: Preferred locations
        tech_stack: Tech stack keywords
        visa_required: Filter for F-1/OPT
        action: Which pipeline stage to run
        auto_approve: Skip human approval gates
    """
    prefs = UserPreferences(
        target_roles=target_roles,
        locations=locations or ["California", "Remote"],
        tech_stack=tech_stack or [],
        visa_required=visa_required,
        resume_text=resume_text,
    )

    initial: AgentState = {
        "messages": [],
        "user_preferences": prefs,
        "requested_action": action,
        "job_listings": [],
        "fit_scores": [],
        "tailored_resumes": [],
        "outreach_drafts": [],
        "current_agent": "",
        "needs_human_approval": False,
        "human_approved": auto_approve,
        "error": "",
    }

    app = compile_workflow()
    return app.invoke(initial)