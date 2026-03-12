"""
Author: Pranay Hedau
Purpose: Pydantic data models, the contracts between all agents.
Every piece of data flowing through the pipeline is defined here.
Agents don't pass raw dicts they pass typed, validated objects.
    If any agent produces malformed data, Pydantic catches it
    immediately instead of letting it silently corrupt downstream agents.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


# ================================================================
# CORE DATA MODELS — one per pipeline stage
# ================================================================
"""Purpose: A single job posting discovered by the Job Scout agent.
    """
class JobListing(BaseModel):
    
    title: str = Field(description="Job title, e.g. 'ML Engineer Intern'")
    company: str = Field(description="Company name, e.g. 'NVIDIA'")
    location: str = Field(default="", description="Job location")
    url: str = Field(default="", description="Application URL")
    summary: str = Field(default="", description="2-3 sentence JD summary")
    key_skills: list[str] = Field(
        default_factory=list,
        description="Required skills extracted from JD"
    )
    posted_date: str = Field(default="", description="When posted")
    visa_friendly: bool = Field(
        default=True,
        description="Whether role supports F-1/OPT sponsorship"
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_none_to_defaults(cls, values):
        """Small LLMs (llama3.2) sometimes return null instead of ""
        for optional string fields. Coerce None → empty string."""
        if isinstance(values, dict):
            str_fields = ["title", "company", "location", "url",
                          "summary", "posted_date"]
            for field in str_fields:
                if field in values and values[field] is None:
                    values[field] = ""
        return values

    @field_validator("key_skills", mode="before")
    @classmethod
    def coerce_key_skills(cls, v):
        """Small LLMs sometimes return key_skills as a single string
        like 'AI/ML Engineering' instead of a list. Coerce to list."""
        if isinstance(v, str):
            return [v] if v else []
        if v is None:
            return []
        return v


"""Purpose: Resume-to-JD match analysis produced by the Fit Analyst agent.

    Contains both the numerical score AND the reasoning (matched skills,
    missing skills, gap analysis).
    """
class FitScore(BaseModel):
    
    job: JobListing  # Reference back to which job this score is for
    overall_score: int = Field(
        ge=0, le=100,
        description="Overall fit percentage (0-100)"
    )
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    gap_analysis: str = Field(
        default="",
        description="Human-readable summary of gaps and recommendations"
    )


"""Purpose: Rewritten resume sections produced by the Resume Tailor agent.

    Doesn't contain the full resume — just the CHANGED parts:
    a new summary, rewritten bullets, and which keywords were added.
    This lets the user see exactly what changed and decide whether
    to accept the modifications.
    """
class TailoredResume(BaseModel):
    
    target_job: JobListing  # Which job this was tailored for
    tailored_summary: str = Field(
        default="",
        description="Rewritten professional summary for this role"
    )
    tailored_bullets: list[str] = Field(
        default_factory=list,
        description="Rewritten experience bullets"
    )
    keywords_added: list[str] = Field(
        default_factory=list,
        description="New JD keywords injected into resume"
    )
    fit_score_before: int = Field(default=0)
    fit_score_after: int = Field(default=0)


"""Purpose: Recruiter outreach messages produced by the Outreach agent.

    Three outputs per job: LinkedIn connection request (200 char limit),
    a comment for a recruiter's hiring post, and a cold email.
    All personalized to the specific role and company.
    """
class OutreachDraft(BaseModel):
    
    job: JobListing  # Which job this outreach targets
    connection_request: str = Field(
        default="",
        description="LinkedIn connection request (max 200 chars)"
    )
    linkedin_comment: str = Field(
        default="",
        description="Comment on recruiter's hiring post"
    )
    cold_email: str = Field(
        default="",
        description="Cold email to hiring manager"
    )
    recruiter_name: str = Field(default="")



# ================================================================
# WORKFLOW CONTROL — how the user tells the system what to do
# ================================================================

"""Purpose: What the user wants the pipeline to do.

    This determines which agents run:
        SEARCH_JOBS   → only Job Scout
        ANALYZE_FIT   → Job Scout → Fit Analyst
        TAILOR_RESUME → Job Scout → Fit Analyst → Resume Tailor
        DRAFT_OUTREACH → all four agents
        FULL_PIPELINE  → all four agents 

    Inheriting from str AND Enum means these values are JSON-serializable.
    """
class AgentAction(str, Enum):
    
    SEARCH_JOBS = "search_jobs"
    ANALYZE_FIT = "analyze_fit"
    TAILOR_RESUME = "tailor_resume"
    DRAFT_OUTREACH = "draft_outreach"
    FULL_PIPELINE = "full_pipeline"


"""Purpose: User input that configures the entire pipeline.

    This is the ENTRY POINT for our system: everything starts with what the user wants.
    The Job Scout uses target_roles and locations to build search queries.
    The Fit Analyst uses resume_text to compare against JDs.
    """
class UserPreferences(BaseModel):
    
    target_roles: list[str] = Field(
        description="Role titles to search, e.g. ['AI ML Intern', 'Backend SWE Intern']"
    )
    locations: list[str] = Field(
        default_factory=lambda: ["California", "Remote"]
    )
    tech_stack: list[str] = Field(
        default_factory=list,
        description="Preferred technologies, e.g. ['Python', 'LangChain', 'AWS']"
    )
    visa_required: bool = Field(
        default=True,
        description="Filter for F-1/OPT friendly roles"
    )
    resume_text: str = Field(
        default="",
        description="Parsed resume content (plain text)"
    )
    resume_path: str = Field(
        default="",
        description="Path to resume PDF file"
    )


# ================================================================
# EVALUATION — measuring agent output quality
# ================================================================
"""Purpose: Single evaluation metric result.

    Used by the evaluation framework to track per-agent quality.
    Results are collected and displayed in the UI's evaluation tab.
    """
class EvalResult(BaseModel):
    
    agent_name: str
    metric_name: str
    score: float = Field(ge=0.0, le=1.0)
    details: str = Field(default="")