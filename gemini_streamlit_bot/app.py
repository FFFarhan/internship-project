import os
from typing import List

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

@st.cache_resource(show_spinner=False)
def load_api_key() -> str:
    """Configure Gemini client once per session and return default model."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env.")
    genai.configure(api_key=api_key)
    return os.getenv("GEMINI_MODEL", "gemini-1.5-pro")


@st.cache_resource(show_spinner=False)
def get_model(model_name: str) -> genai.GenerativeModel:
    return genai.GenerativeModel(model_name)


@st.cache_data(show_spinner=False)
def list_generation_models() -> List[str]:
    try:
        models = list(genai.list_models())
    except Exception as exc:  
        raise RuntimeError(f"Could not retrieve model list: {exc}") from exc
    names = []
    for item in models:
        methods = getattr(item, "supported_generation_methods", [])
        if "generateContent" in methods:
            name = getattr(item, "name", "")
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            if name:
                names.append(name)
    unique_names = list(dict.fromkeys(names)) or []
    return sorted(unique_names)


def ask_gemini(question: str, model_name: str) -> str:
    if not question.strip():
        raise ValueError("Please enter a question before submitting.")
    try:
        model = get_model(model_name)
        response = model.generate_content(question)
    except Exception as exc:  
        raise RuntimeError(f"Gemini API error: {exc}") from exc
    text = getattr(response, "text", "") or ""
    if not text and getattr(response, "candidates", None):
        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                text = "".join(getattr(part, "text", "") for part in parts)
                if text:
                    break
    return text.strip() or "No answer returned."


st.set_page_config(page_title="Gemini AI Q&A Bot", page_icon="✨")
st.title("Gemini AI Q&A Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

try:
    default_model = load_api_key()
except ValueError as err:
    st.error(str(err))
    st.stop()
except Exception as err:  #
    st.error("Failed to initialize Gemini client.")
    st.info(str(err))
    st.stop()

sidebar = st.sidebar
sidebar.header("Model settings")

try:
    available_models = list_generation_models()
except RuntimeError as err:
    sidebar.warning(str(err))
    available_models = []

if default_model not in available_models:
    available_models.insert(0, default_model)
if not available_models:
    available_models = [default_model]

if st.session_state.selected_model is None:
    st.session_state.selected_model = default_model
elif st.session_state.selected_model not in available_models:
    available_models.insert(0, st.session_state.selected_model)

selected_index = 0
if st.session_state.selected_model in available_models:
    selected_index = available_models.index(st.session_state.selected_model)

selected_model = sidebar.selectbox(
    "Gemini model",
    available_models,
    index=selected_index,
    key="selected_model",
    help="Models supporting generateContent are sourced from the Gemini API.",
)

sidebar.caption("Default model from .env: " + default_model + " And" + " Currently selected: " + selected_model + "."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    if not prompt.strip():
        st.warning("Please enter a question before submitting.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_gemini(prompt, selected_model)
                except ValueError as err:
                    st.warning(str(err))
                except RuntimeError as err:
                    st.error(str(err))
                    if "404" in str(err):
                        st.info("Check GEMINI_MODEL in .env. Try gemini-1.5-pro or gemini-1.5-flash.")
                else:
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })
