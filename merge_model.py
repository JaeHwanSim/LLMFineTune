from unsloth import FastLanguageModel
import torch
import os

# 1. 학습된 모델 로드
model_path = "gemma2_neobiotech_adapter"
print(f"Loading adapter from {model_path}...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. 모델 병합 및 저장 (16bit)
# vLLM은 4bit QLoRA 어댑터를 직접 로딩하는 것보다 병합된 모델을 로딩하는 것이 더 안정적이고 빠릅니다.
output_dir = "merged_model_for_vllm"
print(f"Merging model and saving to {output_dir}...")

model.save_pretrained_merged(
    output_dir, 
    tokenizer, 
    save_method = "merged_16bit", # 16bit로 병합 (vLLM에서 bfloat16으로 로드하기 위함)
)

print("Done! You can now serve the model with vLLM.")
