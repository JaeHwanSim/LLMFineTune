from unsloth import FastLanguageModel
import torch
import time
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from datasets import load_dataset

# 1. 모델 및 토크나이저 로드 (Hugging Face 기준)
# 4bit 양자화 모델을 사용하여 3080에서도 여유 있게 실행 가능합니다.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. LoRA 설정 (전문 용어 학습용 가중치 추가)
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # 전문 지식 주입을 위해 약간 높게 설정
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
)

# 3. [Test Before] 학습 전 테스트
print("\n" + "="*30 + "\n[학습 전 답변 확인]\n" + "="*30)
FastLanguageModel.for_inference(model)
prompt_text = "IT-III active Fixture 에 대해서 설명해주고 기존 IT System과의 차이점을 설명해줘"
inputs = tokenizer([f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"], return_tensors = "pt").to("cuda")

# 스트리밍 출력으로 답변 확인
streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = streamer, max_new_tokens = 1024)

# 4. 데이터셋 준비 (한글 KoAlpaca 맛보기)
#dataset = load_dataset("beomi/KoAlpaca-v1.1a", split = "train[:500]") # 100개만 사용

dataset = load_dataset("json", data_files="ProductData/neobiotech_final_1000.jsonl", split="train")

#dataset.save_to_disk("./KoAlpaca-v1.1a.json")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples.get("input", [""] * len(instructions))
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Gemma 2의 공식 프롬프트 템플릿 적용
        text = f"<start_of_turn>user\n{instruction} {input}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)
print(dataset[0:20])

# 5. 학습 (Fine-Tuning)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 8, # CPU 코어도 넉넉할 테니 데이터 처리 속도 UP
    packing = False,
    args = TrainingArguments(
        # [핵심 변경 1] 배치 사이즈 대폭 증가 (VRAM 48GB 활용)
        per_device_train_batch_size = 2,  # 2 -> 8 (메모리 남으면 16까지 도전 가능)
        
        # [핵심 변경 2] 누적 스텝 감소 (업데이트 빈도 증가)
        gradient_accumulation_steps = 8,  # 4 -> 2
        
        # [핵심 변경 3] 4090의 축복 BF16 활성화
        bf16 = True,       # 4090은 무조건 True
        fp16 = False,      # 끄기
        
        num_train_epochs = 5,
        learning_rate = 2e-4,
        logging_steps = 1,
        
        # [선택] 4090은 메모리가 넉넉하므로 굳이 8bit 옵티마이저 안 써도 됩니다.
        # 더 정밀한 학습을 위해 32bit나 paged_adamw_32bit 사용 가능
        # 물론 adamw_8bit를 써도 성능 차이는 거의 없고 VRAM만 아껴줍니다. (그대로 둬도 무방)
        optim = "adamw_8bit", 
        
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        
        # [멀티 GPU] DDP(분산 학습) 관련 설정 (Unsloth가 자동으로 잡지만 명시)
        ddp_find_unused_parameters = False,
    ),
)
start_time = time.time()
trainer_stats = trainer.train()
end_time = time.time()

training_time = end_time - start_time
print(f"\n[Training Time] {training_time // 60:.0f}m {training_time % 60:.0f}s")

# 6. [Test After] 학습 후 테스트
print("\n" + "="*30 + "\n[학습 후 답변 확인]\n" + "="*30)
FastLanguageModel.for_inference(model)
_ = model.generate(**inputs, streamer = streamer, max_new_tokens = 1024)

# 7. 모델 저장 (어댑터만 저장)
model.save_pretrained("gemma2_neobiotech_adapter")
tokenizer.save_pretrained("gemma2_neobiotech_adapter")
print("\n학습된 어댑터가 'gemma2_neobiotech_adapter' 폴더에 저장되었습니다.")