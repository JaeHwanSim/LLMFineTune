from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

# 1. 저장된 모델 로드
model_path = "gemma2_neobiotech_adapter"
print(f"Loading model from {model_path}...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. 추론 모드 설정
FastLanguageModel.for_inference(model)
streamer = TextStreamer(tokenizer)

print("\n" + "="*50)
print("네오바이오텍 AI 챗봇입니다. (종료하려면 'quit' 또는 'exit' 입력)")
print("="*50 + "\n")

while True:
    # 사용자 입력 받기
    user_input = input("질문: ")
    
    if user_input.lower() in ["quit", "exit"]:
        print("챗봇을 종료합니다.")
        break
        
    if not user_input.strip():
        continue

    # 프롬프트 구성
    prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
    
    # 입력 토큰화
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    # 답변 생성 및 출력
    print("\n답변: ", end="")
    _ = model.generate(**inputs, streamer = streamer, max_new_tokens = 512)
    print("\n" + "-"*30 + "\n")
