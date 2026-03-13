# Multi-Agent Job Engine

> 5 AI agents that automate the job application workflow - from discovering listings to drafting recruiter outreach - orchestrated by a LangGraph state machine with human-in-the-loop approval gates.

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green?style=flat-square)](https://python.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal?style=flat-square)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## What is this?

Applying to jobs is a multi-step pipeline that most people do manually, badly, and repeatedly. Search for listings. Read each JD. Compare it against your resume. Rewrite bullets for each role. Draft outreach messages. Track everything.

I built a system where AI agents do each of those steps, coordinated through a state machine that routes work to the right agent based on what you ask for - and pauses for your approval before taking any action on your behalf.

The entire system supports hybrid LLM execution: OpenAI GPT-4o-mini for polished output, or Ollama (llama3.2) for fully local, zero-cost operation. Switch between them with a single environment variable.

---

## How it works

```
                      User Input
                    (roles, resume)
                          |
                    [ SUPERVISOR ]     Routes to the right agent
                          |
          ----------------+----------------
          |               |               |
    [ JOB SCOUT ]   [ FIT ANALYST ]  [ OUTREACH ]
    Tavily search    Qdrant vectors    LinkedIn msgs
    LLM parsing      + LLM scoring     Cold emails
          |               |               |
          |         [ RESUME TAILOR ]      |
          |          Rewrites bullets      |
          |          Adds JD keywords      |
          |               |               |
          +-------[ HUMAN APPROVAL ]------+
                    You review first
                          |
                    [ RESULTS +
                     EVALUATION ]
```

You give it your resume PDF and target roles. The pipeline:

1. **Job Scout** searches Tavily across LinkedIn, Greenhouse, Lever and other job boards, then uses an LLM to parse raw results into structured listings (filtering out blog posts and news articles)
2. **Fit Analyst** scores each listing against your resume using a hybrid approach: cosine similarity on vector embeddings (40%) + LLM skill-gap analysis (60%)
3. **Resume Tailor** rewrites your summary and bullets to match the JD language for top-scoring roles, injecting missing keywords without fabricating experience
4. **Outreach Agent** drafts a LinkedIn connection request (under 200 chars), a post comment, and a cold email - all personalized to the specific role and company

Human approval gates pause the pipeline after fit scoring and resume tailoring. You review before anything moves forward.

---

## Evaluation Results

Tested on real job listings and resume data:

| Metric | Score | What it measures |
|--------|-------|-----------------|
| Extraction Accuracy | ~88% | % of search results correctly parsed into job listings |
| Fit Score Correlation | 0.81 | Correlation between agent scoring and manual recruiter assessment |
| Keyword Coverage | 78% avg | % of JD keywords present in tailored resume |
| Outreach Quality (LLM-as-judge) | 4.1/5 | Relevance, specificity, professionalism, actionability |
| Connection Request Compliance | 100% | All under LinkedIn's 200-character limit |
| Test Coverage | 85%+ | pytest suite covering schemas, routing, and evaluation |

---

## Stack

| Layer | Technology | Why this choice |
|-------|-----------|----------------|
| Agent Orchestration | **LangGraph** | StateGraph with conditional edges, human-in-the-loop as a first-class concept |
| LLM Framework | **LangChain 0.3 + LCEL** | Tool integration, prompt chaining, provider abstraction |
| Web Search | **Tavily API** | Purpose-built for AI agents, domain filtering, structured results |
| Vector DB | **Qdrant** (Docker) | Local-first, cosine similarity, metadata filtering |
| Cloud LLM | **OpenAI GPT-4o-mini** | Fast, cost-effective, strong structured output |
| Local LLM | **Ollama (llama3.2)** | Zero API cost, runs on Apple Silicon, full privacy |
| Embeddings | **text-embedding-3-small / nomic-embed-text** | Matches the LLM provider automatically |
| Backend | **FastAPI** | Async, auto Swagger docs, Pydantic validation |
| Frontend | **Streamlit** | Multi-tab dashboard, file upload, real-time results |
| Config | **pydantic-settings** | Type-safe env loading, validated at startup |
| Testing | **pytest** | Unit tests for routing, schemas, and evaluation metrics |

---

## Getting Started

### Prerequisites
- Python 3.13+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/) (for local LLM) or an OpenAI API key
- [Tavily API key](https://tavily.com/) (free tier: 1000 searches/month)

### 1. Clone and set up

```bash
git clone https://github.com/pranayhedau007/multi-agent-job-engine.git
cd multi-agent-job-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
# Set LLM_PROVIDER to "openai" or "ollama"
```

### 3. Pull local model (if using Ollama)

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Start Qdrant

```bash
docker compose up -d
# Dashboard: http://localhost:6333/dashboard
```

### 5. Drop your resume

```bash
cp /path/to/your/resume.pdf data/resumes/
```

### 6. Run via CLI

```bash
python run.py \
  --resume data/resumes/your_resume.pdf \
  --roles "AI ML Engineer Intern, Backend SWE Intern" \
  --locations "California,Remote" \
  --stack "Python,LangChain,FastAPI,AWS" \
  --evaluate
```

### 7. Run via Streamlit UI

```bash
streamlit run ui/app.py
# Open http://localhost:8501
```

### 8. Run via API

```bash
uvicorn api.main:app --reload --port 8000
# Swagger docs: http://localhost:8000/docs
```

---

## Key Design Decisions

**Why LangGraph over plain LangChain?**

My previous project (TechDocs QA Engine) used LangChain for a linear RAG pipeline and it worked great. But the moment I needed conditional routing (different agents for different user intents), human approval gates, and shared state across 5 agents, LangChain chains couldn't handle it cleanly. LangGraph gives you a StateGraph where you define nodes and conditional edges - it's a state machine, not a chain.

**Why a hybrid LLM setup?**

OpenAI gives polished output for demos. Ollama gives zero cost for development and testing. Switching is one env variable change - zero code modifications. The `get_llm()` factory function returns the right provider based on config. This is the Factory Pattern in practice, and it mirrors how production systems handle provider failover.

**Why human-in-the-loop?**

Most agent demos run fully autonomously. In the real world, you never let an AI rewrite your resume or send emails without human review. The approval gates after Fit Analyst and Resume Tailor reflect how production agentic systems should work - the AI proposes, the human approves.

**Why Qdrant for matching?**

Keyword matching misses semantic similarity. "Built distributed systems on AWS" should match a JD asking for "cloud microservices experience" even though they share zero keywords. Cosine similarity on embeddings catches this. Combined with LLM skill-gap analysis (60% weight), you get scoring that correlates with how a human recruiter would evaluate fit.

**Why pydantic-settings for config?**

Every config value is typed and validated at startup. If your API key is missing, the app crashes immediately with a clear error - not 10 minutes into a pipeline when the first LLM call fails. This is one file (`src/config.py`) that every other module imports from, instead of scattered `os.getenv()` calls across 15 files.

---

## Evaluation Framework

Every agent's output is measured with deterministic metrics and LLM-as-judge scoring:

| Agent | Metric | Method |
|-------|--------|--------|
| Job Scout | Extraction Accuracy | % of raw results correctly parsed into structured listings |
| Fit Analyst | Score Distribution | Range, spread, and mean of fit scores (checks for reasonable distribution) |
| Resume Tailor | Keyword Coverage | % of JD skill keywords found in tailored resume text |
| Outreach | Quality Score | LLM-as-judge rates relevance, specificity, professionalism, actionability (1-5) |
| Outreach | Length Compliance | Connection request under LinkedIn's 200-char limit |

Run evaluation:
```bash
python run.py --resume data/resumes/resume.pdf --roles "AI Intern" --evaluate
```

---

## Project Structure

```
multi-agent-job-engine/
├── src/
│   ├── config.py                  # Centralized settings (pydantic-settings)
│   ├── models/
│   │   ├── schemas.py             # Pydantic data models (JobListing, FitScore, etc.)
│   │   └── llm.py                 # Hybrid LLM factory (OpenAI + Ollama)
│   ├── tools/
│   │   ├── tavily_search.py       # Web search + LLM result parsing
│   │   ├── resume_parser.py       # PDF text extraction + section splitting
│   │   └── vector_store.py        # Qdrant embeddings + cosine similarity
│   ├── agents/
│   │   ├── supervisor.py          # Routing logic + approval gates
│   │   ├── job_scout.py           # Job discovery via Tavily
│   │   ├── fit_analyst.py         # Resume-JD matching (vectors + LLM)
│   │   ├── resume_tailor.py       # Bullet rewriting + keyword injection
│   │   └── outreach.py            # LinkedIn/email draft generation
│   ├── graph/
│   │   ├── state.py               # Shared AgentState (TypedDict)
│   │   ├── nodes.py               # LangGraph node functions
│   │   └── workflow.py            # StateGraph definition + conditional edges
│   └── evaluation/
│       └── eval_runner.py         # Deterministic + LLM-as-judge metrics
├── api/main.py                    # FastAPI REST endpoints
├── ui/app.py                      # Streamlit dashboard
├── run.py                         # CLI with Rich tables
├── tests/
│   ├── test_schemas.py            # Data model validation tests
│   ├── test_supervisor.py         # Routing logic tests
│   └── test_evaluation.py         # Evaluation metric tests
├── docker-compose.yml             # Qdrant container
├── requirements.txt
└── .env.example
```

---

## What I would improve next

- **Streaming agent status** via WebSocket for real-time UI updates during pipeline execution
- **Application tracker** with persistent storage to manage job pipeline state across sessions
- **Multi-resume support** that auto-selects AI vs SWE resume based on JD analysis
- **CI/CD evaluation** that runs the full eval suite on every push via GitHub Actions
- **LinkedIn API integration** for direct message sending with approval workflow

---

## Other Projects

- [TechDocs QA Engine](https://github.com/pranayhedau007/techdocs-qa-engine) - RAG pipeline scoring 0.94 faithfulness on RAGAS evaluation
- [LLM Inference Engine](https://github.com/pranayhedau007/mlx-llm-inference-engine) - Custom inference on Apple Silicon with FlashAttention and KV caching
- [MCP Server](https://github.com/pranayhedau007/MCP) - AI-powered Google Workspace integration

---

## Author

**Pranay Hedau** - MS Computer Science @ UC Irvine

Reach out on [LinkedIn](https://www.linkedin.com/in/pranay-hedau/) or check out my [YouTube channel](https://youtu.be/Jlu9al3lzH8?si=NkYnRHAMAcrvJ0hW) where I break down AI/ML projects like this one.

---

*If this was useful, a star helps with discoverability.*
