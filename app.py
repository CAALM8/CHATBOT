import streamlit as st
import requests

st.set_page_config(page_title="HF Chatbot", page_icon="🤖", layout="wide")
st.title("😊 Hugging Face Chatbot")

# 默认模型
DEFAULT_MODEL = "HuggingFaceH4/zephyr-7b-beta"
API_URL = "https://api-inference.huggingface.co/v1/chat/completions"

# 初始化保存对话
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# 调用 HuggingFace Chat API
# -------------------------------
def hf_chat(messages, token, model):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.7
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        return data["choices"][0]["message"]["content"], None

    except Exception as e:
        return None, f"❌ Error: {e}"


# -------------------------------
# Sidebar 设置
# -------------------------------
with st.sidebar:
    st.header("⚙ 设置")

    token = st.text_input(
        "你的 HuggingFace Token（必填）",
        type="password",
        placeholder="hf_xxxxxxxxx"
    )

    model = st.text_input("模型 ID", DEFAULT_MODEL)

    if st.button("🧹 清空对话"):
        st.session_state.messages = []
        st.success("对话已清空")


# -------------------------------
# 显示历史消息
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------------------
# 输入框
# -------------------------------
prompt = st.chat_input("输入你的问题...")

if prompt:
    if not token:
        st.error("❌ 请先在左侧输入 HuggingFace token")
    else:
        # 保存用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 调用 API
        reply, err = hf_chat(st.session_state.messages, token, model)

        if err:
            with st.chat_message("assistant"):
                st.error(err)
        else:
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.write(reply)
