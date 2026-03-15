"""
Author: Pranay Hedau
Purpose: Evaluation framework - deterministic metrics + LLM-as-judge."""

import json
import logging

from src.models.llm import get_eval_llm
from src.models.schemas import FitScore, TailoredResume, OutreachDraft, EvalResult

logger = logging.getLogger(__name__)


def eval_keyword_coverage(tailored: TailoredResume) -> EvalResult:
    """Purpose: Evaluating by Measuring % of JD keywords present in tailored resume."""
    jd_skills = set(s.lower() for s in tailored.target_job.key_skills)
    if not jd_skills:
        return EvalResult(agent_name="resume_tailor", metric_name="keyword_coverage",
                          score=0.0, details="No JD skills to match")
    all_text = (tailored.tailored_summary + " " + " ".join(tailored.tailored_bullets)).lower()
    matched = sum(1 for s in jd_skills if s in all_text)
    score = matched / len(jd_skills)
    return EvalResult(agent_name="resume_tailor", metric_name="keyword_coverage",
                      score=round(score, 3),
                      details=f"{matched}/{len(jd_skills)} JD keywords in tailored resume")


def eval_connection_length(draft: OutreachDraft) -> EvalResult:
    """Purpose: Check connection request stays under 200 char LinkedIn limit."""
    length = len(draft.connection_request)
    score = 1.0 if length <= 200 else max(0.0, 1.0 - (length - 200) / 100)
    return EvalResult(agent_name="outreach", metric_name="conn_req_length",
                      score=round(score, 3), details=f"{length} chars (limit: 200)")


def eval_outreach_quality(draft: OutreachDraft) -> EvalResult:
    """LLM-as-judge scoring for outreach quality."""
    llm = get_eval_llm()
    prompt = f"""Rate this outreach on 1-5 scale. Return ONLY JSON:
{{"relevance": 4, "specificity": 3, "professionalism": 5, "actionability": 4, "overall": 4.0}}

Connection request: {draft.connection_request}
LinkedIn comment: {draft.linkedin_comment}
Cold email: {draft.cold_email}
Target: {draft.job.title} at {draft.job.company}"""

    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
    if content.endswith("```"):
        content = content[:-3]

    try:
        scores = json.loads(content.strip())
        overall = scores.get("overall", 3.0) / 5.0
        return EvalResult(agent_name="outreach", metric_name="quality_score",
                          score=round(overall, 3), details=json.dumps(scores))
    except Exception:
        return EvalResult(agent_name="outreach", metric_name="quality_score",
                          score=0.0, details="Parse failed")


def eval_fit_distribution(scores: list[FitScore]) -> EvalResult:
    """Purpose: Check fit scores have reasonable spread (not all same value)."""
    if not scores:
        return EvalResult(agent_name="fit_analyst", metric_name="score_distribution",
                          score=0.0, details="No scores")
    vals = [s.overall_score for s in scores]
    spread = max(vals) - min(vals)
    avg = sum(vals) / len(vals)
    score = min(1.0, spread / 50) * 0.5 + (1.0 if 30 < avg < 85 else 0.5) * 0.5
    return EvalResult(agent_name="fit_analyst", metric_name="score_distribution",
                      score=round(score, 3),
                      details=f"Range: {min(vals)}-{max(vals)}, Avg: {avg:.1f}")


def run_evaluation(
    fit_scores: list[FitScore],
    tailored: list[TailoredResume],
    outreach: list[OutreachDraft],
) -> list[EvalResult]:
    """Run all evaluation metrics."""
    results = []
    if fit_scores:
        results.append(eval_fit_distribution(fit_scores))
    for t in tailored:
        results.append(eval_keyword_coverage(t))
    for d in outreach:
        results.append(eval_connection_length(d))
        results.append(eval_outreach_quality(d))
    logger.info(f"Evaluation: {len(results)} metrics computed")
    for r in results:
        logger.info(f"  [{r.agent_name}] {r.metric_name}: {r.score:.3f}")
    return results