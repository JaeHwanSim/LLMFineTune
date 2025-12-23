import os
import json
import glob
from pypdf import PdfReader
from unsloth import FastLanguageModel
from tqdm import tqdm
import re

# 설정
PDF_DIR = "./ProductData"
OUTPUT_FILE = "./ProductData/neobiotech_qa_generated.jsonl"
TARGET_COUNT = 1000
MODEL_NAME = "unsloth/gemma-2-9b-it-bnb-4bit"

# 네오바이오텍 관련 파일만 선택 (타사 제품 제외)
TARGET_FILES = [
    "(2312)NEO CHAIR M5 런칭 안내.pdf",
    "250701_ALX_3단리플릿_ver.1.05.pdf",
    "5-2. 190821 IT-III active 카다록(국문).pdf",
    "ALX 임상증례집 ver. 1 FINAL (1).pdf"
]

def extract_text_from_pdfs(pdf_dir, target_files):
    full_text = ""
    for filename in target_files:
        path = os.path.join(pdf_dir, filename)
        if os.path.exists(path):
            print(f"Reading {filename}...")
            try:
                reader = PdfReader(path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n\n"
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return full_text

def create_chunks(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def generate_qa_pairs(model, tokenizer, chunks, target_count):
    generated_data = []
    
    FastLanguageModel.for_inference(model)
    
    prompt_template = """
    아래 제공된 제품 설명 텍스트를 바탕으로, 고객이 할 법한 질문과 그에 대한 전문적이고 친절한 상담원 스타일의 답변을 생성해주세요.
    
    [조건]
    1. 질문은 구체적이어야 합니다.
    2. 답변은 "네, 고객님." 또는 "문의주신 내용에 대해 설명드리겠습니다."와 같이 상담하는 어조로 시작하며, 풍성하고 상세하게 작성해주세요.
    3. JSON 형식으로 출력해야 합니다. (instruction, input, output 키 사용)
    4. input은 비워두세요 ("").
    5. 가능한 한 많은 QA 쌍을 생성해주세요 (최소 3개 이상).
    
    [텍스트]
    {text}
    
    [출력 형식 예시]
    [
        {{"instruction": "제품의 주요 특징은 무엇인가요?", "input": "", "output": "네, 고객님. 해당 제품의 주요 특징으로는..."}},
        {{"instruction": "가격 정책이 어떻게 되나요?", "input": "", "output": "문의주신 가격 정책에 대해 안내해 드리겠습니다..."}}
    ]
    """
    
    pbar = tqdm(total=target_count)
    
    for chunk in chunks:
        if len(generated_data) >= target_count:
            break
            
        msg = prompt_template.format(text=chunk)
        messages = [
            {"role": "user", "content": msg}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        outputs = model.generate(
            input_ids = inputs,
            max_new_tokens = 2048,
            use_cache = True,
            temperature = 0.7,
            top_p = 0.9,
        )
        
        response = tokenizer.batch_decode(outputs)
        response_text = response[0].split("<start_of_turn>model")[-1].replace("<end_of_turn>", "").strip()
        
        # JSON 파싱 시도
        try:
            # 마크다운 코드 블록 제거
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            # 대괄호 찾기
            start_idx = clean_json.find("[")
            end_idx = clean_json.rfind("]")
            
            if start_idx != -1 and end_idx != -1:
                json_str = clean_json[start_idx:end_idx+1]
                qa_pairs = json.loads(json_str)
                
                for qa in qa_pairs:
                    if "instruction" in qa and "output" in qa:
                        if "input" not in qa:
                            qa["input"] = ""
                        generated_data.append(qa)
                        pbar.update(1)
                        
                        # 실시간 저장 (데이터 유실 방지)
                        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                            json.dump(qa, f, ensure_ascii=False)
                            f.write("\n")
                            
        except json.JSONDecodeError:
            print("JSON parsing failed for a chunk. Skipping...")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue

    pbar.close()
    return generated_data

def main():
    print("1. Loading Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
    )
    
    print("2. Extracting Text from PDFs...")
    text = extract_text_from_pdfs(PDF_DIR, TARGET_FILES)
    print(f"Total extracted text length: {len(text)} characters")
    
    print("3. Chunking Text...")
    chunks = create_chunks(text)
    print(f"Total chunks: {len(chunks)}")
    
    # 기존 파일 초기화
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    print("4. Generating QA Pairs...")
    generate_qa_pairs(model, tokenizer, chunks, TARGET_COUNT)
    
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
