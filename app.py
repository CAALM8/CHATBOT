import gradio as gr
from huggingface_hub import InferenceClient

def chat_with_hf(token, model_id, user_input, history):

    if not token:
        return history + [["You", user_input], ["Bot", "❌ 请先输入你的 Hugging Face Token"]]

    if not model_id:
        return history + [["You", user_input], ["Bot", "❌ 请先输入模型 ID"]]

    try:
        client = InferenceClient(token=token)

        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": user_input}],
            max_tokens=256,
        )

        bot_reply = response.choices[0].message["content"]

        history.append(["你", user_input])
        history.append(["🤖", bot_reply])

        return history

    except Exception as e:
        history.append(["你", user_input])
        history.append(["❌ Error", str(e)])
        return history


with gr.Blocks() as demo:

    gr.Markdown("# 😄 Hugging Face Chatbot")

    with gr.Row():
        token = gr.Textbox(label="你的 HuggingFace Token（必填）", type="password")
        model_id = gr.Textbox(label="模型 ID", placeholder="例如：google/gemma-2b-it")

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(label="你：")
    send_btn = gr.Button("发送")
    clear_btn = gr.Button("清空对话")

    send_btn.click(
        chat_with_hf,
        inputs=[token, model_id, user_input, chatbot],
        outputs=[chatbot]
    )

    clear_btn.click(lambda: None, None, chatbot, queue=False)

demo.launch()
