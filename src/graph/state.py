"""
Author: Pranay Hedau
Purpose: Shared state schema for the LangGraph agent workflow.

This TypedDict is the central data structure that flows through every node.
Each agent reads what it needs and writes its outputs back.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from src.models.schemas import (
    JobListing, FitScore, TailoredResume,
    OutreachDraft, UserPreferences, AgentAction,
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]#why annotated? Cause it tells LangGraph that when new msgs appear, append them in add_messages, to store conversation history
    user_preferences: UserPreferences
    requested_action: AgentAction
    job_listings: list[JobListing]
    fit_scores: list[FitScore]
    tailored_resumes: list[TailoredResume]
    outreach_drafts: list[OutreachDraft]
    current_agent: str
    needs_human_approval: bool
    human_approved: bool
    error: str