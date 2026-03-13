"""
Author: Pranay Hedau
Purpose: LangGraph node functions each wraps a specialist agent."""

import logging
from langchain_core.messages import AIMessage
from src.graph.state import AgentState
from src.agents.job_scout import run_job_scout
from src.agents.fit_analyst import run_fit_analyst
from src.agents.resume_tailor import run_resume_tailor
from src.agents.outreach import run_outreach_agent

logger = logging.getLogger(__name__)


def job_scout_node(state: AgentState) -> dict:
    """Purpose: Search for jobs based on user preferences."""
    logger.info(">>> Job Scout Agent")
    try:
        listings = run_job_scout(state["user_preferences"])
        return {
            "job_listings": listings,
            "current_agent": "job_scout",
            "needs_human_approval": False,
            "messages": [AIMessage(content=f"Found {len(listings)} job listings.")],
        }
    except Exception as e:
        return {
            "job_listings": [], "current_agent": "job_scout",
            "error": str(e),
            "messages": [AIMessage(content=f"Job search error: {e}")],
        }


def fit_analyst_node(state: AgentState) -> dict:
    """Purpose: Analyze resume fit against discovered jobs."""
    logger.info(">>> Fit Analyst Agent")
    listings = state.get("job_listings", [])
    if not listings:
        return {
            "fit_scores": [], "current_agent": "fit_analyst",
            "messages": [AIMessage(content="No listings to analyze.")],
        }
    try:
        scores = run_fit_analyst(state["user_preferences"].resume_text, listings)
        top3 = "\n".join(
            f"  {s.job.company} - {s.job.title}: {s.overall_score}/100"
            for s in scores[:3]
        )
        return {
            "fit_scores": scores,
            "current_agent": "fit_analyst",
            "needs_human_approval": True,
            "messages": [AIMessage(content=f"Fit analysis done. Top matches:\n{top3}")],
        }
    except Exception as e:
        return {
            "fit_scores": [], "current_agent": "fit_analyst",
            "error": str(e),
            "messages": [AIMessage(content=f"Fit analysis error: {e}")],
        }


def resume_tailor_node(state: AgentState) -> dict:
    """Purpose: Tailor resume for top-matching jobs."""
    logger.info(">>> Resume Tailor Agent")
    fit_scores = state.get("fit_scores", [])
    if not fit_scores:
        return {
            "tailored_resumes": [], "current_agent": "resume_tailor",
            "messages": [AIMessage(content="No fit scores available.")],
        }
    try:
        tailored = run_resume_tailor(state["user_preferences"].resume_text, fit_scores)
        summary = "\n".join(
            f"  {t.target_job.company}: {t.fit_score_before} -> {t.fit_score_after}"
            for t in tailored
        )
        return {
            "tailored_resumes": tailored,
            "current_agent": "resume_tailor",
            "needs_human_approval": True,
            "messages": [AIMessage(content=f"Tailored for {len(tailored)} roles:\n{summary}")],
        }
    except Exception as e:
        return {
            "tailored_resumes": [], "current_agent": "resume_tailor",
            "error": str(e),
            "messages": [AIMessage(content=f"Tailoring error: {e}")],
        }


def outreach_node(state: AgentState) -> dict:
    """Purpose: Draft outreach messages for tailored roles."""
    logger.info(">>> Outreach Agent")
    tailored = state.get("tailored_resumes", [])
    if not tailored:
        return {
            "outreach_drafts": [], "current_agent": "outreach",
            "messages": [AIMessage(content="No tailored resumes for outreach.")],
        }
    try:
        drafts = run_outreach_agent(tailored, state["user_preferences"].resume_text[:500])
        return {
            "outreach_drafts": drafts,
            "current_agent": "outreach",
            "messages": [AIMessage(content=f"Outreach drafts ready for {len(drafts)} roles. Done!")],
        }
    except Exception as e:
        return {
            "outreach_drafts": [], "current_agent": "outreach",
            "error": str(e),
            "messages": [AIMessage(content=f"Outreach error: {e}")],
        }


def human_approval_node(state: AgentState) -> dict:
    """Purpose: Placeholder for human-in-the-loop approval gate."""
    return {"needs_human_approval": True, "human_approved": False}