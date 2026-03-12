"""
Author: Pranay Hedau
Purpose: Outreach Agent drafts personalized LinkedIn/email messages per job."""

import json
import logging

from src.models.llm import get_llm
from src.models.schemas import TailoredResume, OutreachDraft

logger = logging.getLogger(__name__)


def draft_outreach(tailored: TailoredResume, resume_summary: str) -> OutreachDraft:
    """Purpose: Generate outreach drafts for a single job."""
    llm = get_llm(temperature=0.5)
    job = tailored.target_job

    prompt = f"""Career coach writing outreach for an MS CS student at UC Irvine with 4+ years backend
experience at Barclays, pivoting to AI/ML. Has Salesforce certs and AI/ML YouTube channel.

TARGET: {job.title} at {job.company} ({job.location})
Skills needed: {', '.join(job.key_skills)}
Tailored summary: {tailored.tailored_summary}
Resume highlights: {resume_summary[:500]}

Return ONLY valid JSON:
{{"connection_request": "Under 200 chars. Short, specific, not generic.",
"linkedin_comment": "3-5 sentences on a hiring post. Company-specific.",
"cold_email": "Subject + body. 4-6 sentences. Lead with relevance, mention one project/metric.",
"recruiter_name": ""}}

No generic phrases. Reference something specific about the company. Confident tone."""

    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
    if content.endswith("```"):
        content = content[:-3]

    try:
        p = json.loads(content.strip())
        cr = p.get("connection_request", "")
        if len(cr) > 200:
            cr = cr[:197] + "..."
        return OutreachDraft(
            job=job, connection_request=cr,
            linkedin_comment=p.get("linkedin_comment", ""),
            cold_email=p.get("cold_email", ""),
            recruiter_name=p.get("recruiter_name", ""),
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Outreach parse failed: {e}")
        return OutreachDraft(job=job)


def run_outreach_agent(
    tailored_resumes: list[TailoredResume], resume_summary: str
) -> list[OutreachDraft]:
    """Purpose: Generate outreach drafts for all tailored resume targets."""
    logger.info(f"Outreach Agent: drafting for {len(tailored_resumes)} roles")
    return [draft_outreach(t, resume_summary) for t in tailored_resumes]