"""
Author: Pranay Hedau
Purpose: Resume Tailor Agent rewrites resume bullets to match target JD language."""

import json
import logging

from src.models.llm import get_llm
from src.models.schemas import FitScore, TailoredResume

logger = logging.getLogger(__name__)


def tailor_for_job(resume_text: str, fit: FitScore) -> TailoredResume:
    """Purpose: Rewrite resume sections to maximize fit for a specific job."""
    llm = get_llm(temperature=0.4)
    job = fit.job
    jd = (
        f"Title: {job.title}\nCompany: {job.company}\n"
        f"Skills: {', '.join(job.key_skills)}\nSummary: {job.summary}"
    )

    prompt = f"""Expert resume writer. Rewrite resume for this JD. Return ONLY valid JSON:
{{"tailored_summary": "2-3 sentences targeting this role",
"tailored_bullets": ["bullet1", "bullet2", "bullet3", "bullet4", "bullet5"],
"keywords_added": ["kw1", "kw2", "kw3"],
"estimated_new_score": 85}}

RULES: Never fabricate experience. Rephrase using JD language. Add JD keywords naturally. Start bullets with action verbs.

RESUME:
{resume_text[:3000]}

TARGET JOB:
{jd}

FIT: {fit.overall_score}/100 | Matched: {', '.join(fit.matched_skills)} | Missing: {', '.join(fit.missing_skills)}"""

    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
    if content.endswith("```"):
        content = content[:-3]

    try:
        p = json.loads(content.strip())
        return TailoredResume(
            target_job=job,
            tailored_summary=p.get("tailored_summary", ""),
            tailored_bullets=p.get("tailored_bullets", []),
            keywords_added=p.get("keywords_added", []),
            fit_score_before=fit.overall_score,
            fit_score_after=p.get("estimated_new_score", fit.overall_score + 10),
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Tailor failed: {e}")
        return TailoredResume(
            target_job=job, fit_score_before=fit.overall_score,
            fit_score_after=fit.overall_score
        )


def run_resume_tailor(
    resume_text: str, fit_scores: list[FitScore], top_n: int = 5
) -> list[TailoredResume]:
    """Purpose: Tailor resume for top-N jobs scoring 50+."""
    eligible = [f for f in fit_scores if f.overall_score >= 50][:top_n]
    logger.info(f"Resume Tailor: processing {len(eligible)} jobs")
    results = [tailor_for_job(resume_text, f) for f in eligible]
    for t in results:
        logger.info(
            f"  {t.target_job.company}: {t.fit_score_before} -> "
            f"{t.fit_score_after} (+{len(t.keywords_added)} keywords)"
        )
    return results