"""
Author: Pranay Hedau
Purpose: Resume parser — extracts text from PDF and splits into sections.

Two functions:
    parse_resume()     → PDF file path in, full text out
    extract_sections() → full text in, dict of sections out

Why m I introducing two steps instead of one?
    Some agents need the FULL text (Fit Analyst compares entire resume
    against a JD). Other agents need SPECIFIC sections (Resume Tailor
    only rewrites the summary and experience bullets, not education).
    Splitting lets each agent grab what it needs.

Usage:
    from src.tools.resume_parser import parse_resume, extract_sections

    text = parse_resume("data/resumes/my_resume.pdf")
    sections = extract_sections(text)
    print(sections["skills"])  # just the skills section
Date Created: 03-07-2026    
"""

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


"""Purpose: Extract all text content from a resume PDF.

    Args:
        file_path: Path to the PDF file

    Returns:
        Full text as a single string, pages joined by newlines

    Raises:
        FileNotFoundError: If the PDF doesn't exist
        ValueError: If the file isn't a PDF
    """
def parse_resume(file_path: str) -> str:
    
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Resume not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    # pdfplumber opens the PDF and extracts text page by page
    # Why page-by-page? Because some pages might fail (scanned images,
    # corrupted pages) and we don't want one bad page to kill the whole parse
    text_parts = []

    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                logger.warning(f"Page {i + 1} returned no text (possibly scanned/image)")

    full_text = "\n".join(text_parts).strip()

    if not full_text:
        logger.warning(f"No text extracted from {file_path} — may be a scanned PDF")
        return ""

    logger.info(f"Parsed resume: {len(full_text)} chars from {len(text_parts)} page(s)")
    return full_text


"""Purpose: Split resume text into named sections.

    Looks for common resume section headers (SUMMARY, SKILLS, EXPERIENCE, etc.)
    and splits the text at each one. Everything before the first header
    goes into a "header" section (usually name + contact info).

    Args:
        resume_text: Full resume text from parse_resume()

    Returns:
        Dict mapping section names to their content, e.g.:
        {
            "header": "PRANAY HEDAU\\nIrvine, CA | ...",
            "summary": "MS CS at UC Irvine...",
            "skills": "Python, Java, C++...",
            "experience": "Barclays — BA4 Software Developer...",
            "projects": "TechDocs QA Engine...",
            "education": "University of California, Irvine..."
        }

    Why did I not use an LLM for this?
        LLM calls cost money and add latency. Section headers in resumes
        follow predictable patterns. Simple string matching works for 90%
        of resumes and is instant. We save LLM calls for tasks that
        actually need intelligence (like gap analysis and tailoring).
    """
def extract_sections(resume_text: str) -> dict[str, str]:
    
    # These headers cover most standard resume formats
    # Lowercase for case-insensitive matching
    section_headers = [
        "summary", "objective", "profile",
        "skills", "technical skills",
        "experience", "work experience", "professional experience",
        "projects", "notable projects",
        "education",
        "certifications", "certifications & recognition",
    ]

    lines = resume_text.split("\n")
    sections: dict[str, str] = {}
    current_section = "header"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip().lower()

        # Check if this line is a section header
        matched_header = None
        for header in section_headers:
            if stripped == header or stripped == header + ":":
                matched_header = header
                break

        if matched_header:
            # Save the previous section before starting a new one
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = matched_header
            current_lines = []
        else:
            current_lines.append(line)

    # For the last section
    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    logger.info(f"Extracted {len(sections)} sections: {list(sections.keys())}")
    return sections