import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Hugging Face Chatbot", layout="wide")
st.title("😀 Hugging Face Chatbot (Streamlit)")

# Sidebar settings
st.sidebar.header("设置")
HF_TOKEN = st.sidebar.text_input("你的 HuggingFace Token（必填）", type="password")
MODEL_ID = st.sidebar.text_input("模型 ID", "Qwen/Qwen2.5-7B-Instruct")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("请输入消息...")

if user_input and HF_TOKEN:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("正在生成..."):

            try:
                client = InferenceClient(
                    model=MODEL_ID,
                    token=HF_TOKEN
                )

                # 使用统一的 Chat Completions API (最稳)
                response = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["messages"]
                    ],
                    max_tokens=200,
                    temperature=0.7,
                )

                reply = response.choices[0].message["content"]
                st.session_state["messages"].append({"role": "assistant", "content": reply})
                st.write(reply)

            except Exception as e:
                st.error(f"❌ 发生错误：{str(e)}")
