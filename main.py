"""
AI Candidate Screening Agent
----------------------------
A Streamlit UI powered by LangChain + LangGraph + OpenAI.

Run:
    pip install -r requirements.txt
    # Set OPENAI_API_KEY in your environment (or enter it in the sidebar)
    streamlit run main.py
"""

import os
import re
from typing import TypedDict

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


# ---------------------------- Page configuration ----------------------------

st.set_page_config(
    page_title="AI Candidate Screening",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------- Styling ----------------------------------

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px; }
      .hero {
        padding: 1.5rem 1.7rem; border-radius: 16px;
        background: linear-gradient(120deg, #14243a, #245b75);
        color: white; margin-bottom: 1.3rem;
      }
      .hero h1 { color: white; margin: 0 0 .35rem 0; }
      .hero p { color: #e5eef5; margin: 0; font-size: 1.02rem; }
      .result-card {
        padding: 1rem 1.15rem; border: 1px solid #dce3ea;
        border-radius: 12px; min-height: 120px;
      }
      .result-label { color: #637386; font-size: .83rem; text-transform: uppercase;
                      letter-spacing: .06em; margin-bottom: .35rem; }
      .result-value { font-size: 1.25rem; font-weight: 650; }
      .muted { color: #687789; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------- State ------------------------------------

class CandidateState(TypedDict, total=False):
    application: str
    experience_level: str
    skill_match: str
    response: str


# ----------------------------- Agent functions ------------------------------

def clean_label(value: str, allowed: tuple[str, ...], fallback: str) -> str:
    """Normalize an LLM response to one of the expected routing labels."""
    text = re.sub(r"[^a-zA-Z -]", "", value or "").strip().lower()
    for label in allowed:
        if text == label.lower() or label.lower() in text:
            return label
    return fallback


def build_screening_graph(api_key: str, model_name: str):
    """Build and compile the candidate-screening LangGraph."""
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
    )
    parser = StrOutputParser()

    experience_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an HR screening assistant. Classify experience level using "
                "the candidate's application. Return exactly one label: "
                "Entry-level, Mid-level, or Senior-level. Do not add explanation.",
            ),
            ("human", "Candidate application:\n{application}"),
        ]
    )

    skills_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are screening for a Python Developer role. Assess whether the "
                "application demonstrates relevant Python development skills. "
                "Return exactly Match or No Match. Do not add explanation.",
            ),
            ("human", "Candidate application:\n{application}"),
        ]
    )

    experience_chain = experience_prompt | llm | parser
    skills_chain = skills_prompt | llm | parser

    def categorize_experience(state: CandidateState) -> CandidateState:
        raw = experience_chain.invoke({"application": state["application"]})
        return {
            "experience_level": clean_label(
                raw,
                ("Entry-level", "Mid-level", "Senior-level"),
                "Mid-level",
            )
        }

    def assess_skillset(state: CandidateState) -> CandidateState:
        raw = skills_chain.invoke({"application": state["application"]})
        return {
            "skill_match": clean_label(raw, ("Match", "No Match"), "No Match")
        }

    def schedule_hr_interview(state: CandidateState) -> CandidateState:
        return {
            "response": "Candidate has been shortlisted for an HR interview."
        }

    def escalate_to_recruiter(state: CandidateState) -> CandidateState:
        return {
            "response": (
                "Candidate has senior-level experience but does not match the "
                "Python Developer skill requirements. Escalate to the recruiter "
                "for manual review."
            )
        }

    def reject_application(state: CandidateState) -> CandidateState:
        return {
            "response": (
                "Candidate does not meet the job requirements and has been "
                "rejected by the screening workflow."
            )
        }

    def route_application(state: CandidateState) -> str:
        if state["skill_match"] == "Match":
            return "schedule_hr_interview"
        if state["experience_level"] == "Senior-level":
            return "escalate_to_recruiter"
        return "reject_application"

    workflow = StateGraph(CandidateState)
    workflow.add_node("categorize_experience", categorize_experience)
    workflow.add_node("assess_skillset", assess_skillset)
    workflow.add_node("schedule_hr_interview", schedule_hr_interview)
    workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
    workflow.add_node("reject_application", reject_application)

    workflow.add_edge(START, "categorize_experience")
    workflow.add_edge("categorize_experience", "assess_skillset")
    workflow.add_conditional_edges(
        "assess_skillset",
        route_application,
        {
            "schedule_hr_interview": "schedule_hr_interview",
            "escalate_to_recruiter": "escalate_to_recruiter",
            "reject_application": "reject_application",
        },
    )
    workflow.add_edge("schedule_hr_interview", END)
    workflow.add_edge("escalate_to_recruiter", END)
    workflow.add_edge("reject_application", END)

    return workflow.compile()


def run_candidate_screening(application: str, api_key: str, model_name: str):
    """Run the graph and return the candidate screening results."""
    graph = build_screening_graph(api_key, model_name)
    return graph.invoke({"application": application})


# ---------------------------------- UI --------------------------------------

st.markdown(
    """
    <div class="hero">
      <h1>🧑‍💼 AI Candidate Screening Agent</h1>
      <p>Review a candidate application, assess experience and Python skills,
      and route the application to the appropriate next step.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input(
        "OpenAI API key",
        value="",
        type="password",
        help="Alternatively, set OPENAI_API_KEY as an environment variable.",
    )
    model_name = st.selectbox(
        "OpenAI model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
    )
    st.caption("Your API key is used for this session and is not displayed.")

api_key = api_key_input.strip() or os.getenv("OPENAI_API_KEY", "").strip()

left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.subheader("Candidate application")
    with st.form("screening_form"):
        application_text = st.text_area(
            "Paste the candidate's resume summary or application",
            height=260,
            placeholder=(
                "Example: I have 4 years of software development experience, "
                "including 3 years building Python applications, REST APIs, "
                "SQL integrations and automated tests..."
            ),
        )
        submitted = st.form_submit_button(
            "Run candidate screening",
            type="primary",
            use_container_width=True,
        )

with right:
    st.subheader("How the workflow works")
    st.markdown(
        """
        1. **Categorize experience** — Entry-level, Mid-level or Senior-level.
        2. **Assess Python skills** — Match or No Match.
        3. **Route the application**:
           - Match → HR interview
           - No Match + Senior-level → Recruiter review
           - Otherwise → Reject
        """
    )
    st.info(
        "This is an AI-assisted demonstration. A recruiter should review "
        "the application and all screening outcomes before making a hiring decision."
    )

if submitted:
    if not application_text.strip():
        st.warning("Please paste a candidate application before running screening.")
    elif not api_key:
        st.error(
            "OpenAI API key not found. Enter it in the sidebar or set "
            "the OPENAI_API_KEY environment variable."
        )
    else:
        with st.spinner("Analyzing the application and running the workflow..."):
            try:
                result = run_candidate_screening(
                    application_text.strip(),
                    api_key,
                    model_name,
                )
                st.session_state["screening_result"] = result
                st.session_state["screening_application"] = application_text.strip()
            except Exception as exc:
                st.error(f"Screening could not be completed: {exc}")

if "screening_result" in st.session_state:
    result = st.session_state["screening_result"]
    st.divider()
    st.subheader("Screening results")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Experience level</div>
              <div class="result-value">{result["experience_level"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Python skill assessment</div>
              <div class="result-value">{result["skill_match"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Workflow decision")
    response = result["response"]
    if "shortlisted" in response.lower():
        st.success(response)
    elif "escalate" in response.lower():
        st.warning(response)
    else:
        st.error(response)

    with st.expander("View submitted application"):
        st.write(st.session_state["screening_application"])

    with st.expander("View raw workflow output"):
        st.json(
            {
                "experience_level": result["experience_level"],
                "skill_match": result["skill_match"],
                "response": result["response"],
            }
        )

st.divider()
st.markdown(
    '<p class="muted">AI Candidate Screening Agent · LangChain · LangGraph · Streamlit</p>',
    unsafe_allow_html=True,
)
