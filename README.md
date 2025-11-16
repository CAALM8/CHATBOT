
# Free Chatbot (Hugging Face Inference) - TinyLlama Demo

This is a demo Streamlit chatbot that uses Hugging Face Inference API to call an open-source chat model.

## Quick start

1. Create a Hugging Face account and get an access token (https://huggingface.co/settings/tokens).

2. In Streamlit Cloud, add the token to **Secrets** (or locally create `.streamlit/secrets.toml`):

```
HF_TOKEN = "hf_your_token_here"
```

3. Deploy to Streamlit Cloud or run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes and limitations

- The recommended model is: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (used in this demo).
- Hugging Face free inference has limited quota. Heavy usage may be rate-limited or charged.
- This app constructs a simple prompt from the session history — you can improve formatting or use structured chat formats as needed.
