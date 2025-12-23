from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from datasets import load_dataset

# 1. 모델 및 토크나이저 로드 (Hugging Face 기준)
# 4bit 양자화 모델을 사용하여 3080에서도 여유 있게 실행 가능합니다.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b-it-bnb-4bit",
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

dataset = load_dataset("json", data_files="ProductData/neobiotech_qa_100.jsonl", split="train")

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
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 10,
        #max_steps = 40, # 3080 기준 약 3~5분 소요
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)
trainer.train()

# 6. [Test After] 학습 후 테스트
print("\n" + "="*30 + "\n[학습 후 답변 확인]\n" + "="*30)
FastLanguageModel.for_inference(model)
_ = model.generate(**inputs, streamer = streamer, max_new_tokens = 1024)

# 7. 모델 저장 (어댑터만 저장)
model.save_pretrained("gemma2_neobiotech_adapter")
tokenizer.save_pretrained("gemma2_neobiotech_adapter")
print("\n학습된 어댑터가 'gemma2_neobiotech_adapter' 폴더에 저장되었습니다.")