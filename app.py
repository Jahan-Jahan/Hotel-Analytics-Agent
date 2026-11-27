"""Streamlit demo UI.   streamlit run app.py"""
import pandas as pd
import streamlit as st

from hotel_agent import ui_helpers as ui
from hotel_agent.agent import build_agent
from hotel_agent.config import ConfigError, Settings
from hotel_agent.tools import ToolBox

st.set_page_config(page_title="Hotel Analytics Agent", page_icon="🏨", layout="wide")

# Each paragraph picks its own direction, so Persian answers render right-to-left.
st.markdown("""<style>
.stChatMessage p, .stChatMessage li { unicode-bidi: plaintext; text-align: start; }
</style>""", unsafe_allow_html=True)

settings = Settings.from_env()


@st.cache_resource
def get_toolbox() -> ToolBox:
    return ToolBox(settings)          # shared DB connection pool


def render_call(call: dict) -> None:
    icon = "✅" if call["ok"] else "❌"
    label = {"run_sql": "SQL query", "predict_cancellation": "ML prediction", "make_chart": "Chart"}.get(
        call["tool"], call["tool"])
    st.markdown(f"**{icon} {label}** · {call['seconds']} s")
    if call.get("sql"):
        st.code(call["sql"], language="sql")
    if call.get("inputs"):
        st.json(call["inputs"], expanded=False)
    if not call["ok"]:
        st.error(call["error"])
    elif call["tool"] == "run_sql":
        st.dataframe(pd.DataFrame(call["rows"], columns=call["columns"]), hide_index=True, width="stretch")
        if call["truncated"]:
            st.caption("Result truncated to the row limit.")


def render_assistant(msg: dict) -> None:
    st.markdown(msg["content"])
    for call in msg["calls"]:
        if call["tool"] == "predict_cancellation" and call["ok"]:
            st.metric("Cancellation probability", f"{call['probability']:.1%}")
            st.progress(min(max(call["probability"], 0.0), 1.0))
            verdict = "likely to cancel" if call["will_cancel"] else "likely to stay"
            st.caption(f"Model verdict at threshold {call['threshold']:.2f}: **{verdict}**. "
                       f"This is a statistical estimate, not a certainty.")
    for path in msg["charts"]:
        st.image(path)
    if msg["calls"]:
        with st.expander(f"How I got this · {len(msg['calls'])} tool call(s)"):
            for call in msg["calls"]:
                render_call(call)
    stats = msg["stats"]
    st.caption(f"{stats['steps']} step(s) · {stats['tool_calls']} tool call(s) · "
               f"{stats['tokens']:,} tokens · {stats['seconds']} s")
    if msg.get("error"):
        st.warning(f"Note: {msg['error']}")


# ------------------------------------------------------------------ session
if "agent" not in st.session_state:
    try:
        st.session_state.agent = build_agent(settings, get_toolbox())
    except ConfigError as exc:
        st.error(f"{exc}")
        st.stop()
    st.session_state.messages = []
    st.session_state.tokens = 0
    st.session_state.pending = None

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🏨 Hotel Analytics Agent")
    st.caption("Every number comes from a read-only SQL query or the trained ML model.")
    st.subheader("Services")
    for name, (ok, message) in ui.check_services(settings, get_toolbox()).items():
        st.markdown(f"{'🟢' if ok else '🔴'} **{name}**: {message}")
    st.subheader("Try asking")
    for question in ui.EXAMPLE_QUESTIONS:
        if st.button(question, key=f"ex_{question[:40]}", width="stretch"):
            st.session_state.pending = question
    st.divider()
    st.caption(f"Session tokens: {st.session_state.tokens:,}")
    if st.button("Clear conversation", width="stretch"):
        st.session_state.agent.reset()
        st.session_state.messages = []
        st.session_state.tokens = 0
        st.rerun()

# ------------------------------------------------------------------ chat
st.title("Ask your hotel data")
st.caption("English or فارسی. Questions about bookings, cancellations, revenue, or the risk of a specific booking.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_assistant(msg) if msg["role"] == "assistant" else st.markdown(msg["content"])

prompt = st.chat_input("Ask about the bookings…")
if prompt is None and st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Querying the data…"):
                result = st.session_state.agent.ask(prompt)
        except Exception as exc:                       # never show a stack trace in a demo
            st.error(f"Something went wrong: {type(exc).__name__}: {exc}")
            st.stop()
        message = ui.make_payload(result)
        render_assistant(message)
    st.session_state.messages.append(message)
    st.session_state.tokens += message["stats"]["tokens"]
