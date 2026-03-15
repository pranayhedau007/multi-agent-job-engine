"""
Author: Pranay Hedau
Purpose: Streamlit UI - interactive dashboard for the multi-agent job engine."""

import streamlit as st
from src.tools.resume_parser import parse_resume
from src.models.schemas import AgentAction
from src.graph.workflow import run_pipeline
from src.evaluation.eval_runner import run_evaluation

st.set_page_config(page_title="Multi-Agent Job Engine", page_icon="🎯", layout="wide")
st.title("🎯 Multi-Agent Job Engine")
st.caption("Search jobs -> Analyze fit -> Tailor resume -> Draft outreach")

# Sidebar config
with st.sidebar:
    st.header("Configuration")
    uploaded = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    roles = st.text_input("Target Roles (comma-separated)", "AI ML Engineer Intern, Backend SWE Intern")
    locations = st.text_input("Locations", "California, Remote")
    tech_stack = st.text_input("Tech Stack", "Python, LangChain, FastAPI, AWS")
    visa = st.checkbox("F-1/OPT Visa Required", value=True)
    action = st.selectbox("Action", [
        "Full Pipeline", "Search Jobs Only", "Analyze Fit", "Tailor Resume", "Draft Outreach"
    ])
    auto_approve = st.checkbox("Auto-approve (skip human gates)", value=True)

action_map = {
    "Full Pipeline": "full_pipeline", "Search Jobs Only": "search_jobs",
    "Analyze Fit": "analyze_fit", "Tailor Resume": "tailor_resume",
    "Draft Outreach": "draft_outreach",
}

# Parse resume
resume_text = ""
if uploaded:
    path = f"/tmp/{uploaded.name}"
    with open(path, "wb") as f:
        f.write(uploaded.getvalue())
    resume_text = parse_resume(path)
    with st.sidebar:
        st.success(f"Resume parsed: {len(resume_text)} chars")

# Run
if st.button("Run Pipeline", type="primary", use_container_width=True):
    if not resume_text:
        st.error("Upload a resume first.")
    else:
        role_list = [r.strip() for r in roles.split(",") if r.strip()]
        loc_list = [l.strip() for l in locations.split(",") if l.strip()]
        stack_list = [s.strip() for s in tech_stack.split(",") if s.strip()]

        with st.spinner("Agents working... (1-2 minutes)"):
            result = run_pipeline(
                resume_text=resume_text, target_roles=role_list,
                locations=loc_list, tech_stack=stack_list,
                visa_required=visa, action=AgentAction(action_map[action]),
                auto_approve=auto_approve,
            )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Jobs", "Fit Scores", "Tailored Resume", "Outreach", "Evaluation"
        ])

        with tab1:
            jobs = result.get("job_listings", [])
            st.subheader(f"Found {len(jobs)} Jobs")
            for j in jobs:
                with st.expander(f"{j.title} at {j.company} - {j.location}"):
                    st.write(j.summary)
                    st.write(f"**Skills:** {', '.join(j.key_skills)}")
                    if j.url:
                        st.markdown(f"[Apply]({j.url})")

        with tab2:
            fits = result.get("fit_scores", [])
            st.subheader(f"Fit Analysis ({len(fits)} jobs)")
            for f in fits:
                icon = "🟢" if f.overall_score >= 70 else "🟡" if f.overall_score >= 50 else "🔴"
                with st.expander(f"{icon} {f.job.company} - {f.job.title}: {f.overall_score}/100"):
                    st.write(f"**Matched:** {', '.join(f.matched_skills)}")
                    st.write(f"**Missing:** {', '.join(f.missing_skills)}")
                    st.write(f"**Gaps:** {f.gap_analysis}")

        with tab3:
            tailored = result.get("tailored_resumes", [])
            st.subheader(f"Tailored for {len(tailored)} Roles")
            for t in tailored:
                with st.expander(f"{t.target_job.company}: {t.fit_score_before} -> {t.fit_score_after}"):
                    st.write(f"**Summary:** {t.tailored_summary}")
                    for bullet in t.tailored_bullets:
                        st.write(f"- {bullet}")
                    st.write(f"**Keywords Added:** {', '.join(t.keywords_added)}")

        with tab4:
            drafts = result.get("outreach_drafts", [])
            st.subheader(f"Outreach ({len(drafts)} roles)")
            for d in drafts:
                with st.expander(f"{d.job.company} - {d.job.title}"):
                    st.write(f"**Connection Request** ({len(d.connection_request)} chars):")
                    st.code(d.connection_request)
                    st.write("**LinkedIn Comment:**")
                    st.write(d.linkedin_comment)
                    st.write("**Cold Email:**")
                    st.write(d.cold_email)

        with tab5:
            st.subheader("Evaluation Metrics")
            with st.spinner("Evaluating..."):
                evals = run_evaluation(
                    result.get("fit_scores", []),
                    result.get("tailored_resumes", []),
                    result.get("outreach_drafts", []),
                )
            for e in evals:
                st.metric(f"[{e.agent_name}] {e.metric_name}", f"{e.score:.2f}")
                st.caption(e.details)