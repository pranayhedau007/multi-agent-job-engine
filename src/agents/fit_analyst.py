"""
Author: Pranay Hedau
Purpose: Fit Analyst Agent scores resume-JD match using embeddings + LLM analysis."""

import json
import logging

from src.models.llm import get_llm
from src.models.schemas import JobListing, FitScore
from src.tools.vector_store import compute_similarity

logger = logging.getLogger(__name__)


def _llm_skill_analysis(resume_text: str, job: JobListing) -> dict:
    """Purpose: Use LLM to extract matched/missing skills and gap analysis."""
    llm = get_llm(temperature=0.1)
    jd = (
        f"Title: {job.title}\nCompany: {job.company}\n"
        f"Summary: {job.summary}\nSkills: {', '.join(job.key_skills)}"
    )

    prompt = f"""Analyze resume-job fit. Return ONLY valid JSON, no explanation:
{{"matched_skills": ["skill1", "skill2"], "missing_skills": ["skill1"], "fit_percentage": 75, "gap_analysis": "2-3 sentence summary"}}

Score guide: 90-100 perfect, 70-89 good, 50-69 moderate, below 50 weak.

RESUME:
{resume_text[:3000]}

JOB:
{jd}"""

    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
    if content.endswith("```"):
        content = content[:-3]

    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM skill analysis")
        return {
            "matched_skills": [], "missing_skills": [],
            "fit_percentage": 0, "gap_analysis": "Analysis failed"
        }


def analyze_single_job(resume_text: str, job: JobListing) -> FitScore:
    """Purpose: Analyze fit for a single job listing.
    
    Combines two signals:
        - Vector similarity (40%): semantic match between resume and JD
        - LLM analysis (60%): structured skill gap assessment
    """
    jd_text = f"{job.title} {job.summary} {' '.join(job.key_skills)}"
    semantic_score = compute_similarity(resume_text, jd_text)
    analysis = _llm_skill_analysis(resume_text, job)

    llm_score = analysis.get("fit_percentage", 0)
    combined = max(0, min(100, int(semantic_score * 100 * 0.4 + llm_score * 0.6)))

    return FitScore(
        job=job,
        overall_score=combined,
        matched_skills=analysis.get("matched_skills", []),
        missing_skills=analysis.get("missing_skills", []),
        gap_analysis=analysis.get("gap_analysis", ""),
    )


def run_fit_analyst(resume_text: str, listings: list[JobListing]) -> list[FitScore]:
    """Purpose: Score all job listings against resume, sorted by fit descending."""
    logger.info(f"Fit Analyst: scoring {len(listings)} jobs")
    scores = [analyze_single_job(resume_text, job) for job in listings]
    scores.sort(key=lambda s: s.overall_score, reverse=True)
    for s in scores[:5]:
        logger.info(f"  {s.job.company} - {s.job.title}: {s.overall_score}/100")
    return scores