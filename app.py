import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="HF Chatbot", layout="wide")

st.title("😄 Hugging Face Chatbot (Streamlit)")

# --- Sidebar ---
st.sidebar.header("设置")

token = st.sidebar.text_input("你的 HuggingFace Token（必填）", type="password")
model_id = st.sidebar.text_input("模型 ID", value="google/gemma-2b-it")

if "history" not in st.session_state:
    st.session_state.history = []

# --- Chat input ---
user_input = st.chat_input("输入你的消息...")

if user_input:
    if not token:
        st.error("❌ 请先在左侧输入 HuggingFace Token")
    else:
        client = InferenceClient(token=token)

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": user_input}],
                max_tokens=256,
            )

            reply = response.choices[0].message["content"]

            st.session_state.history.append(("user", user_input))
            st.session_state.history.append(("bot", reply))

        except Exception as e:
            st.error(f"❌ Error: {e}")

# --- Display chat history ---
for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
