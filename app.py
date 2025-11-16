import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Hugging Face Chatbot", layout="wide")

st.title("😀 Hugging Face Chatbot (Streamlit)")

# --- Sidebar settings ---
st.sidebar.header("设置")

HF_TOKEN = st.sidebar.text_input("你的 HuggingFace Token（必填）", type="password")
MODEL_ID = st.sidebar.text_input("模型 ID", "Qwen/Qwen2.5-7B-Instruct")

# 如果你想测试别的模型，填几个可用的预设：
# - meta-llama/Llama-3.1-8B-Instruct
# - mistralai/Mistral-Nemo-Instruct-2407
# - google/gemma-2-2b-it
# - Qwen/Qwen2.5-7B-Instruct


# --- Chat UI ---
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

                # 调用 HF Inference API（自动选择文本生成模型）
                response = client.text_generation(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.7,
                )

                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.write(response)

            except Exception as e:
                st.error(f"❌ 发生错误：{str(e)}")
