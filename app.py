import streamlit as st
import requests
import time

st.set_page_config(page_title="Free Chatbot (Hugging Face)", layout="wide")

st.title("💬 Free Chatbot — Hugging Face Inference API (TinyLlama)")

st.markdown(
    """
This demo uses the Hugging Face Inference API to call an open-source chat model.\n
**Important:** Put your Hugging Face API token in Streamlit secrets as `HF_TOKEN`.\n
Model (recommended): `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
"""
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

if "history" not in st.session_state:
    # history will be list of dicts: {"role": "user" or "bot", "content": "..."}
    st.session_state.history = []
if "system" not in st.session_state:
    st.session_state.system = "You are a helpful assistant."

with st.sidebar:
    st.header("Settings")
    st.text_input("System prompt (role)", value=st.session_state.system, key="system_input")
    if st.button("Set system prompt"):
        st.session_state.system = st.session_state.system_input
        st.session_state.history = []
    st.markdown("---")
    st.markdown("Hugging Face token must be saved in Streamlit secrets as `HF_TOKEN`.\n\nExample `secrets.toml`:\n```\nHF_TOKEN = \"hf_xxx...\"\n```")
    st.markdown("---")
    st.write("Model: " + MODEL_ID)
    st.markdown("Note: Hugging Face free inference has limited quota. Use sparingly.")

def hf_generate(prompt, token, model_id=MODEL_ID, max_length=512):
    """Call Hugging Face Inference API text generation endpoint."""
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "return_full_text": False},
        "options": {"wait_for_model": True}
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Many models return a list of generations with 'generated_text'
        if isinstance(data, dict) and data.get("error"):
            return None, f"Model error: {data.get('error')}"
        if isinstance(data, list) and len(data) > 0:
            # Some HF responses are like [{"generated_text": "..."}]
            text = data[0].get("generated_text") or data[0].get("generated_text", "")
            return text, None
        # fallback if API uses different schema
        return str(data), None
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error: {e} - {resp.text if 'resp' in locals() else ''}"
    except requests.exceptions.Timeout:
        return None, "Request timed out."
    except Exception as e:
        return None, f"Request failed: {e}"

st.subheader("Chat")

cols = st.columns([4,1])
with cols[0]:
    user_input = st.text_input("You:", key="user_input")
with cols[1]:
    send = st.button("Send")

# Display history
for msg in st.session_state.history:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    if role == "user":
        st.markdown(f"**You:** {content}")
    else:
        st.markdown(f"**Bot:** {content}")

if send and user_input.strip():
    # Append user message
    st.session_state.history.append({"role":"user", "content": user_input.strip()})
    # Build prompt: simple concatenation of system + messages
    prompt_parts = [f"System: {st.session_state.system}", "Conversation:"]
    for m in st.session_state.history:
        prefix = "User" if m["role"]=="user" else "Assistant"
        prompt_parts.append(f"{prefix}: {m['content']}")
    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)

    # Get token from secrets
    hf_token = None
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        st.error("Hugging Face token not found in Streamlit secrets. Please set HF_TOKEN in secrets.toml")
    if hf_token:
        with st.spinner("Generating..."):
            text, err = hf_generate(prompt, hf_token)
            if err:
                st.error(err)
            else:
                # Append assistant reply and re-render
                st.session_state.history.append({"role":"bot", "content": text})
                st.experimental_rerun()
