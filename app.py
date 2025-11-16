import streamlit as st
from huggingface_hub import InferenceClient

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Hugging Face Chatbot (Streamlit)",
    layout="wide",
)

st.title("🤖 Hugging Face Chatbot (Streamlit)")

# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.header("Settings")

# Token input
hf_token = st.sidebar.text_input(
    "Your HuggingFace Token (required)", 
    type="password",
    help="Paste your HuggingFace Access Token here."
)

# Model input
model_id = st.sidebar.text_input(
    "Model ID",
    value="Qwen/Qwen2.5-7B-Instruct",
    help="Example: Qwen/Qwen2.5-7B-Instruct"
)

# Token Tutorial (English)
with st.sidebar.expander("How to Get Your HuggingFace Token?"):
    st.markdown("""
### 🔑 How to Get Your HuggingFace Token

1. Go to **https://huggingface.co/settings/tokens**
2. Click **'New token'**
3. Choose **Read Access** (for public models)
4. Copy the generated token
5. Paste it into the field above

⚠️ **Do NOT share your token publicly!**
    """)

st.sidebar.markdown("---")

# -----------------------------
# Initialize Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display Chat History
# -----------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# -----------------------------
# User Input
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input and hf_token and model_id:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    try:
        # -----------------------------
        # HuggingFace Inference Client
        # -----------------------------
        client = InferenceClient(api_key=hf_token)

        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": user_input}]
        )

        assistant_reply = completion.choices[0].message["content"]

        # Save reply
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        st.chat_message("assistant").write(assistant_reply)

    except Exception as e:
        st.error(f"Error: {e}")

elif user_input and not hf_token:
    st.error("❗ Please enter your HuggingFace token in the left sidebar.")
