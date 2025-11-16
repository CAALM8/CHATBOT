import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Hugging Face Chatbot", layout="wide")

st.title("😀 Hugging Face Chatbot (Streamlit)")

# Sidebar
st.sidebar.header("设置")
HF_TOKEN = st.sidebar.text_input("你的 HuggingFace Token（必填）", type="password")
MODEL_ID = st.sidebar.text_input("模型 ID", "Qwen/Qwen2.5-7B-Instruct")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("请输入消息...")

if prompt and HF_TOKEN:
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("正在生成回应..."):

            try:
                client = InferenceClient(
                    model=MODEL_ID,
                    token=HF_TOKEN
                )

                # --- 使用 conversational（适配 Qwen 系列） ---
                response = client.conversational(
                    messages=st.session_state["messages"],
                    max_new_tokens=256,
                    temperature=0.7,
                )

                reply = response["generated_text"]

                st.session_state["messages"].append({"role": "assistant", "content": reply})
                st.write(reply)

            except Exception as e:
                st.error(f"❌ 发生错误：{str(e)}")
