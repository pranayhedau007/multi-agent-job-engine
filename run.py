"""
Author : Pranay Hedau
Purpose: CLI runner for all my fellow devs - quick way to test the full pipeline from terminal."""

import argparse
import logging

from rich.console import Console
from rich.table import Table

from src.tools.resume_parser import parse_resume
from src.models.schemas import AgentAction
from src.graph.workflow import run_pipeline
from src.evaluation.eval_runner import run_evaluation

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Job Engine CLI")
    parser.add_argument("--resume", required=True, help="Path to resume PDF")
    parser.add_argument("--roles", required=True, help="Comma-separated target roles")
    parser.add_argument("--locations", default="California,Remote")
    parser.add_argument("--stack", default="", help="Comma-separated tech stack")
    parser.add_argument("--action", default="full_pipeline",
                        choices=[a.value for a in AgentAction])
    parser.add_argument("--no-visa", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    console.print("\n[bold cyan]Multi-Agent Job Engine[/]\n")

    resume_text = parse_resume(args.resume)
    console.print(f"[green]Resume parsed:[/] {len(resume_text)} chars\n")

    roles = [r.strip() for r in args.roles.split(",")]
    locs = [l.strip() for l in args.locations.split(",")]
    stack = [s.strip() for s in args.stack.split(",") if s.strip()]

    result = run_pipeline(
        resume_text=resume_text, target_roles=roles, locations=locs,
        tech_stack=stack, visa_required=not args.no_visa,
        action=AgentAction(args.action), auto_approve=True,
    )

    # Jobs table
    jobs = result.get("job_listings", [])
    if jobs:
        t = Table(title=f"Jobs Found ({len(jobs)})")
        t.add_column("Company", style="cyan")
        t.add_column("Title")
        t.add_column("Location")
        for j in jobs:
            t.add_row(j.company, j.title, j.location)
        console.print(t)

    # Fit scores
    fits = result.get("fit_scores", [])
    if fits:
        t = Table(title="Fit Scores")
        t.add_column("Company", style="cyan")
        t.add_column("Title")
        t.add_column("Score", justify="right")
        t.add_column("Matched Skills")
        for f in fits[:10]:
            color = "green" if f.overall_score >= 70 else "yellow" if f.overall_score >= 50 else "red"
            t.add_row(f.job.company, f.job.title,
                      f"[{color}]{f.overall_score}/100[/]",
                      ", ".join(f.matched_skills[:5]))
        console.print(t)

    # Tailored
    for tr in result.get("tailored_resumes", []):
        console.print(f"\n[bold]Tailored for {tr.target_job.company}[/] "
                      f"({tr.fit_score_before} -> {tr.fit_score_after})")
        console.print(f"  Summary: {tr.tailored_summary[:200]}")

    # Outreach
    for d in result.get("outreach_drafts", []):
        console.print(f"\n[bold]Outreach: {d.job.company}[/]")
        console.print(f"  Connection: {d.connection_request}")

    # Evaluation
    if args.evaluate:
        console.print("\n[bold yellow]Running Evaluation...[/]")
        evals = run_evaluation(
            fits, result.get("tailored_resumes", []),
            result.get("outreach_drafts", []),
        )
        t = Table(title="Evaluation")
        t.add_column("Agent")
        t.add_column("Metric")
        t.add_column("Score", justify="right")
        t.add_column("Details")
        for e in evals:
            t.add_row(e.agent_name, e.metric_name, f"{e.score:.3f}", e.details[:80])
        console.print(t)

    console.print("\n[bold green]Done![/]\n")


if __name__ == "__main__":
    main()