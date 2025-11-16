import streamlit as st
import requests
import json

# -----------------------------
# Streamlit UI 设置
# -----------------------------
st.set_page_config(page_title="HF Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Hugging Face Chatbot")

# 默认模型
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# session_state 初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system" not in st.session_state:
    st.session_state.system = "You are a helpful assistant."

# -----------------------------
# 🔥 Hugging Face 新版 API 封装函数
# -----------------------------
def hf_generate(prompt, token, model_id=MODEL_ID):
    """
    使用 Hugging Face Inference Router 新接口：
    https://router.huggingface.co/hf-inference/chat/completions
    """
    api_url = "https://router.huggingface.co/hf-inference/chat/completions"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": st.session_state.system},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()  # 若失败直接抛出
        data = resp.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"], None
        else:
            return None, f"⚠ Unexpected response: {data}"

    except Exception as e:
        return None, f"❌ Request error: {e}"


# -----------------------------
# Sidebar 设置
# -----------------------------
with st.sidebar:
    st.header("⚙ 设置")

    token = st.text_input(
        "你的 HuggingFace Token（必填）",
        type="password",
        placeholder="hf_xxxxxxxxxxxxx"
    )

    st.session_state.system = st.text_area(
        "系统提示词 System Prompt",
        st.session_state.system
    )

    MODEL_ID = st.text_input(
        "模型 ID（可选）",
        MODEL_ID
    )

    if st.button("🧹 清空对话"):
        st.session_state.messages = []
        st.success("已清空！")


# -----------------------------
# 显示历史消息
# -----------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])


# -----------------------------
# 输入框：用户输入
# -----------------------------
prompt = st.chat_input("请输入你的问题…")

if prompt and token:
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 调用 HF API
    reply, err = hf_generate(prompt, token, MODEL_ID)

    if err:
        reply = err

    # 显示模型回复
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)

elif prompt and not token:
    st.error("❌ 请先在左侧填入 HuggingFace API Token！")
