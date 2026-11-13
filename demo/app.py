"""
Streamlit chat UI for NovaBot.

Run from the project root:   streamlit run demo/app.py

The sidebar shows which document chunks were retrieved for the last question, so you
can SEE when the poisoned vendor_update.txt or customer_records.txt gets pulled in.
On Day 5 we add a "Firewall ON / OFF" switch here.
"""

import streamlit as st

from rag_app import NovaBot

st.set_page_config(page_title="NovaBot (vulnerable)", page_icon="🛒")
st.title("NovaBot — NovaMart support")
st.caption("A deliberately vulnerable RAG bot. Firewall: OFF")


# st.cache_resource builds the bot once and reuses it across reruns.
# (Streamlit re-runs the whole script on every interaction.)
@st.cache_resource
def get_bot() -> NovaBot:
    return NovaBot()


bot = get_bot()

# Chat history lives in session_state so it survives reruns.
if "history" not in st.session_state:
    st.session_state.history = []
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask NovaBot something (or try an attack)…")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = bot.ask(question)
                answer = result.answer
                st.session_state.last_retrieved = result.retrieved
            except Exception as e:  # e.g. missing API key, rate limit
                answer = f"Error calling the LLM: {e}"
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.subheader("Retrieved for last question")
    if not st.session_state.last_retrieved:
        st.write("Nothing yet.")
    for chunk in st.session_state.last_retrieved:
        st.markdown(f"**{chunk.source}**")
        st.caption(chunk.text[:300] + ("…" if len(chunk.text) > 300 else ""))
    if st.button("Clear chat"):
        st.session_state.history = []
        st.session_state.last_retrieved = []
        st.rerun()
