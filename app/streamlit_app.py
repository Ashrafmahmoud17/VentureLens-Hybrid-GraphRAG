import os

import requests
import streamlit as st


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


st.set_page_config(
    page_title="VentureLens",
    page_icon="🔎",
    layout="wide",
)


st.title("🔎 VentureLens")
st.caption("Hybrid GraphRAG Career & Skill Intelligence")

# Sidebar
with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Retrieval Results (Top K)",
        min_value=1,
        max_value=20,
        value=5,
    )

    st.divider()

    st.subheader("Backend Status")

    try:
        health_response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )

        if health_response.ok:
            st.success("API Connected")
        else:
            st.error("API Unavailable")

    except requests.RequestException:
        st.error("Cannot connect to API")

    st.caption(f"API: {API_BASE_URL}")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def display_retrieval(retrieval):
    counts = retrieval.get("counts", {})

    st.markdown("#### Retrieval Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Vector", counts.get("vector", 0))
    col2.metric("Keyword", counts.get("keyword", 0))
    col3.metric("Fused", counts.get("fused", 0))

    st.caption(
        f"Method: {retrieval.get('retrieval_method', 'Unknown')}"
    )

    results = retrieval.get("results", [])

    with st.expander(f"Retrieved Nodes ({len(results)})"):
        for index, item in enumerate(results, start=1):
            node_type = item.get("node_type", "Unknown")
            node_id = item.get("node_id", "Unknown")

            st.markdown(
                f"**{index}. {node_type} — {node_id}**"
            )

            st.write(item.get("document", ""))

            st.caption(
                f"Sources: {', '.join(item.get('sources', []))}"
            )

            neighbors = item.get("neighbors", [])

            if neighbors:
                with st.expander(
                    f"Graph Connections ({len(neighbors)})",
                    expanded=False,
                ):
                    for neighbor in neighbors:
                        properties = neighbor.get("properties", {})

                        st.write({
                            "relationship": neighbor.get("relationship"),
                            "direction": neighbor.get("direction"),
                            "labels": neighbor.get("labels"),
                            "properties": properties,
                        })

            if index < len(results):
                st.divider()


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            retrieval = message.get("retrieval")

            if retrieval:
                display_retrieval(retrieval)


# Chat input
question = st.chat_input(
    "Ask about jobs, skills, courses, or projects..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge graph and generating answer..."):

            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/ask",
                    json={
                        "question": question,
                        "top_k": top_k,
                    },
                    timeout=240,
                )

                response.raise_for_status()
                result = response.json()

                answer = result.get("answer", "No answer returned.")
                retrieval = result.get("retrieval", {})

                st.markdown(answer)

                display_retrieval(retrieval)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "retrieval": retrieval,
                })

            except requests.RequestException as exc:
                error_message = f"API request failed: {exc}"

                if getattr(exc, "response", None) is not None:
                    try:
                        error_message = exc.response.json().get(
                            "detail",
                            error_message,
                        )
                    except ValueError:
                        pass

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {error_message}",
                })

            except (ValueError, KeyError) as exc:
                error_message = f"Invalid API response: {exc}"

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                })
