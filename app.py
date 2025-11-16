import streamlit as st
import requests

st.set_page_config(page_title="HF Chatbot", layout="wide")
st.title("😄 Hugging Face Chatbot")

# Sidebar
st.sidebar.header("⚙ 设置")
token = st.sidebar.text_input("你的 HuggingFace Token（必填）", type="password")
model_id = st.sidebar.text_input("模型 ID", value="HuggingFaceH4/zephyr-7b-beta")

if "messages" not in st.session_state:
    st.session_state.messages = []

def hf_chat(messages, model, token):
    url = "https://router.huggingface.co/hf-inference/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 256,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)

# Chat UI
user_input = st.text_input("✏️ 你:", "")

if st.button("发送"):
    if not token:
        st.error("❗ 请在左侧输入 Hugging Face Token")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

        res, err = hf_chat(st.session_state.messages, model_id, token)

        if err:
            st.error(f"❌ Error: {err}")
        else:
            reply = res["choices"][0]["message"]["content"]
            st.session_state.messages.append({"role": "assistant", "content": reply})

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑 **你:** {msg['content']}")
    else:
        st.markdown(f"🤖 **AI:** {msg['content']}")

if st.sidebar.button("🧹 清空对话"):
    st.session_state.messages = []
    st.rerun()
