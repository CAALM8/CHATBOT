import streamlit as st
import requests
import json

st.set_page_config(page_title="HF Chatbot", layout="wide")
st.title("😄 Hugging Face Chatbot")

# Sidebar
st.sidebar.header("⚙ 设置")
token = st.sidebar.text_input("你的 HuggingFace Token（必填）", type="password")
model_id = st.sidebar.text_input("模型 ID", value="HuggingFaceH4/zephyr-7b-beta")

if "messages" not in st.session_state:
    st.session_state.messages = []

def hf_chat(messages, model, token):
    # HF 官方 Inference API（正确可用）
    url = f"https://api-inference.huggingface.co/models/{model}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # 将消息压缩成 prompt（HF text-generation 接口不支持 messages）
    prompt = ""
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        prompt += f"{role}: {m['content']}\n"
    prompt += "Assistant:"

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 300, "return_full_text": False}
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # HuggingFace 输出格式： [{"generated_text": "..."}]
        reply = data[0]["generated_text"]
        return reply, None

    except Exception as e:
        return None, str(e)

# UI
user_input = st.text_input("✏️ 你:")

if st.button("发送"):
    if not token:
        st.error("❗ 请先输入 HuggingFace Token")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

        reply, err = hf_chat(st.session_state.messages, model_id, token)

        if err:
            st.error(f"❌ Error: {err}")
        else:
            st.session_state.messages.append({"role": "assistant", "content": reply})

# 显示消息
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑 **你:** {msg['content']}")
    else:
        st.markdown(f"🤖 **AI:** {msg['content']}")

if st.sidebar.button("🧹 清空对话"):
    st.session_state.messages = []
    st.rerun()
