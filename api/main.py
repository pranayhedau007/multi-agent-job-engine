"""
Author: Pranay Hedau
Purpose: FastAPI backend - REST endpoints for the agent pipeline."""

import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.models.schemas import AgentAction
from src.tools.resume_parser import parse_resume
from src.graph.workflow import run_pipeline
from src.evaluation.eval_runner import run_evaluation

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Multi-Agent Job Engine",
    description="AI-powered job search, resume tailoring, and outreach generation",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class PipelineRequest(BaseModel):
    target_roles: list[str]
    locations: list[str] = ["California", "Remote"]
    tech_stack: list[str] = []
    visa_required: bool = True
    resume_text: str = ""
    action: str = "full_pipeline"


@app.get("/health")
def health():
    return {"status": "ok", "service": "multi-agent-job-engine"}


@app.post("/run")
def run_agents(req: PipelineRequest):
    if not req.resume_text:
        raise HTTPException(400, "resume_text is required")
    action_map = {a.value: a for a in AgentAction}
    action = action_map.get(req.action, AgentAction.FULL_PIPELINE)

    result = run_pipeline(
        resume_text=req.resume_text, target_roles=req.target_roles,
        locations=req.locations, tech_stack=req.tech_stack,
        visa_required=req.visa_required, action=action, auto_approve=True,
    )
    return {
        "jobs": [j.model_dump() for j in result.get("job_listings", [])],
        "fit_scores": [f.model_dump() for f in result.get("fit_scores", [])],
        "tailored_resumes": [t.model_dump() for t in result.get("tailored_resumes", [])],
        "outreach_drafts": [o.model_dump() for o in result.get("outreach_drafts", [])],
        "error": result.get("error", ""),
    }


@app.post("/evaluate")
def evaluate(req: PipelineRequest):
    if not req.resume_text:
        raise HTTPException(400, "resume_text is required")
    result = run_pipeline(
        resume_text=req.resume_text, target_roles=req.target_roles,
        locations=req.locations, tech_stack=req.tech_stack,
        visa_required=req.visa_required, action=AgentAction.FULL_PIPELINE,
        auto_approve=True,
    )
    evals = run_evaluation(
        result.get("fit_scores", []),
        result.get("tailored_resumes", []),
        result.get("outreach_drafts", []),
    )
    return {"evaluations": [e.model_dump() for e in evals]}


@app.post("/parse-resume")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    text = parse_resume(path)
    return {"filename": file.filename, "text": text, "length": len(text)}