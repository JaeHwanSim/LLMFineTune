import gradio as gr
from openai import OpenAI

# vLLM 서버 설정
API_URL = "http://localhost:8000/v1"
MODEL_NAME = "gemma-3-finetuned"

print(f"Connecting to vLLM at {API_URL}...")
client = OpenAI(
    base_url=API_URL,
    api_key="EMPTY"
)

def chat_function(message, history, system_message):
    messages = [{"role": "system", "content": system_message}]
    
    # history is list of [user_msg, assistant_msg]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1024
        )

        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                partial_message += content
                yield partial_message
                
    except Exception as e:
        yield f"오류가 발생했습니다: {str(e)}\n\nvLLM 서버가 실행 중인지 확인해주세요."

def user_input(user_message, history):
    # Gradio 6.0 Chatbot defaults to type="messages"
    # history is list of dicts: [{'role': 'user', 'content': '...'}, ...]
    return "", history + [{"role": "user", "content": user_message}]

def bot_response(history, system_message):
    # history[-1] is the user message dict
    user_message = history[-1]['content']
    
    # Construct messages for API call
    # Convert Gradio history (dicts) to API messages
    # We need to exclude the last user message from history for the API call context if we want to process it separately,
    # but chat_function expects the full history including the current message?
    # Actually chat_function expects history as list of [user, bot] tuples in previous version.
    # Let's rewrite chat_function to accept list of dicts directly.
    
    # Re-using chat_function logic but adapted for list of dicts
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history) # history already includes the last user message
    
    # Add a placeholder for assistant response
    history.append({"role": "assistant", "content": ""})
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1024
        )

        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                partial_message += content
                history[-1]['content'] = partial_message
                yield history
                
    except Exception as e:
        history[-1]['content'] = f"오류가 발생했습니다: {str(e)}"
        yield history

with gr.Blocks(title="Neobiotech AI Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦷 Neobiotech AI 상담 챗봇")
    gr.Markdown(f"모델: `{MODEL_NAME}` (vLLM Serving)")
    
    with gr.Accordion("고급 설정", open=False):
        system_prompt_input = gr.Textbox(
            value="당신은 유능한 AI 비서입니다.",
            label="시스템 프롬프트",
            lines=2
        )

    chatbot = gr.Chatbot(height=600)
    msg = gr.Textbox(label="메시지 입력", placeholder="질문을 입력하세요...", show_label=False)
    submit_btn = gr.Button("전송")
    clear_btn = gr.Button("대화 지우기")

    # Event wiring
    msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot, system_prompt_input], chatbot
    )
    submit_btn.click(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot, system_prompt_input], chatbot
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
