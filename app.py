"""Streamlit UI for the AI Travel Planning Assistant."""

import os
import sys
from pathlib import Path

# Ensure project root is on path when launched from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="AI Travel Planning Assistant",
    page_icon="✈️",
    layout="wide",
)

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.agent.assistant import ask
from src.config import DEFAULT_LLM_MODEL, DESTINATION
from src.rag.ingestion import build_vector_store
from src.rag.retriever import ensure_vector_store

SAMPLE_QUESTIONS = [
    "What are the must-visit attractions in Singapore?",
    "How can a tourist travel around Singapore?",
    "What is the weather forecast for the next 3 days?",
    "Convert INR 50,000 to SGD.",
    "Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast.",
    "I have a budget of INR 60,000. Convert it to SGD and suggest a three-day itinerary.",
    "Suggest activities for a family with children.",
    "What indoor attractions can I visit if rain is expected?",
]


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []


@st.cache_resource(show_spinner="Loading knowledge base (first run may take 1–2 minutes)...")
def _cached_vector_store():
    return ensure_vector_store()


def ensure_kb():
    try:
        _cached_vector_store()
    except Exception as exc:
        st.error(f"Knowledge base failed to load: {exc}")
        st.stop()


def main():
    init_session()

    st.title("✈️ AI Travel Planning Assistant")
    st.caption(
        f"Destination: **{DESTINATION}** | RAG + MCP tools | LLM: OpenAI (`{DEFAULT_LLM_MODEL}`)"
    )

    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            This assistant combines:
            - **RAG** — Singapore travel guides (attractions, transport, itineraries)
            - **MCP Weather** — live forecast via Open-Meteo
            - **MCP Currency** — live rates via Frankfurter API
            """
        )

        st.header("Sample Questions")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, key=f"sample_{q[:30]}", use_container_width=True):
                st.session_state.pending_question = q

        st.divider()
        if st.button("Rebuild Knowledge Base", use_container_width=True):
            _cached_vector_store.clear()
            with st.spinner("Rebuilding vector store..."):
                build_vector_store()
            st.success("Knowledge base rebuilt.")
            st.rerun()

        st.divider()
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("Add `OPENAI_API_KEY` to `.env` to enable AI responses.")
        else:
            st.success("OpenAI API key loaded.")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    ensure_kb()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metadata"):
                with st.expander("Sources & Tools Used"):
                    meta = message["metadata"]
                    if meta.get("kb_sources"):
                        st.markdown("**Knowledge Base Sources:**")
                        for src in meta["kb_sources"]:
                            if src.get("url"):
                                st.markdown(f"- [{src['title']}]({src['url']})")
                            else:
                                st.markdown(f"- {src['title']}")
                    if meta.get("mcp_tools_used"):
                        st.markdown("**MCP Tools Used:**")
                        for tool_name in meta["mcp_tools_used"]:
                            st.markdown(f"- {tool_name}")
                    if not meta.get("kb_sources") and not meta.get("mcp_tools_used"):
                        st.markdown("No external sources retrieved.")

    prompt = st.chat_input("Ask about Singapore travel, weather, or budget...")
    if "pending_question" in st.session_state:
        prompt = st.session_state.pop("pending_question")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Planning your trip..."):
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    response = ask(prompt, history=history)
                    st.markdown(response.answer)

                    metadata = {
                        "kb_sources": response.kb_sources,
                        "mcp_tools_used": response.mcp_tools_used,
                        "used_rag": response.used_rag,
                    }
                    with st.expander("Sources & Tools Used"):
                        if response.kb_sources:
                            st.markdown("**Knowledge Base Sources:**")
                            for src in response.kb_sources:
                                if src.get("url"):
                                    st.markdown(f"- [{src['title']}]({src['url']})")
                                else:
                                    st.markdown(f"- {src['title']}")
                        if response.mcp_tools_used:
                            st.markdown("**MCP Tools Used:**")
                            for tool_name in response.mcp_tools_used:
                                st.markdown(f"- {tool_name}")
                except Exception as exc:
                    response_text = f"Sorry, I encountered an error: {exc}"
                    st.error(response_text)
                    metadata = {}
                    response = type("R", (), {"answer": response_text})()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.answer if hasattr(response, "answer") else response_text,
                "metadata": metadata,
            }
        )


main()
