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

                # 拼接历史对话，构成 prompt
                history_text = ""
                for m in st.session_state["messages"]:
                    role = "User" if m["role"] == "user" else "Assistant"
                    history_text += f"{role}: {m['content']}\n"
                history_text += "Assistant:"

                # 直接使用 text_generation —— 所有版本都支持！
                output = client.text_generation(
                    prompt=history_text,
                    max_new_tokens=200,
                    temperature=0.7
                )

                reply = output
                st.session_state["messages"].append({"role": "assistant", "content": reply})
                st.write(reply)

            except Exception as e:
                st.error(f"❌ 发生错误：{str(e)}")
