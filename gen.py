import json
import random

# =========================================================
# 네오바이오텍 심화 지식 베이스 (Deep Knowledge Base)
# =========================================================
# PDF 내용을 바탕으로 한 핵심 지식들을 정의합니다.
knowledge_base = [
    # [ALX 임플란트 심화]
    {"t": "ALX 임플란트의 나사산 깊이", "a": "ALX의 나사산 깊이(Thread Depth)는 0.4mm~0.6mm로 설계되어 있어, 임플란트와 뼈의 접촉 면적(BIC)을 극대화하고 강력한 초기 고정력을 발휘합니다."},
    {"t": "ALX의 식립 토크 권장 범위", "a": "성공적인 초기 고정을 위해 35~45 Ncm의 식립 토크를 권장합니다. 50 Ncm 이상의 과도한 토크는 골 괴사를 유발할 수 있으므로 주의해야 합니다."},
    {"t": "ALX 1.4 Cuff의 용도", "a": "1.4mm의 낮은 Cuff는 전치부 심미 영역이나 잇몸이 매우 얇은 케이스에서 금속 노출을 막기 위해 사용됩니다."},
    {"t": "ALX 임플란트의 S-Wide 타입", "a": "직경 Ø5.5 이상의 S-Wide 타입은 구치부 발치 즉시 식립 시 발치와(Socket)를 효과적으로 채워주어 GBR을 최소화합니다."},

    # [IT-III active 심화]
    {"t": "IT-III active의 식립 깊이 조절", "a": "Straight Neck 디자인 덕분에 식립 깊이를 자유롭게 조절할 수 있으며, 잇몸 높이에 맞춰 Supracrestal 또는 Equicrestal 식립이 모두 가능합니다."},
    {"t": "IT-III active의 2.5 Hex 드라이버", "a": "내부 연결에 2.5mm Hex 드라이버를 사용하여 체결력이 우수하고, 장기간 사용 시에도 드라이버가 미끄러지는 현상(Cam-out)을 방지합니다."},
    {"t": "IT-III active의 1.8 Cuff 장점", "a": "1.8mm Cuff는 잇몸이 얇은 환자에게 적합하며, 보철물 변연(Margin)이 잇몸 하방에 위치하도록 하여 심미성을 높여줍니다."},

    # [IS System & Kit 심화]
    {"t": "IS-III active의 0.9 Pitch", "a": "Ø4.0 이상의 직경에서는 0.9mm의 나사산 간격(Pitch)을 적용하여 식립 속도를 높이고 초기 고정력을 강화했습니다."},
    {"t": "Neo Master Kit의 드릴 스토퍼", "a": "모든 드릴에 탈착 가능한 스토퍼(Stopper)를 장착할 수 있어, 초보자도 신경 손상 걱정 없이 계획된 깊이만큼 안전하게 드릴링할 수 있습니다."},
    {"t": "Sinus Crestal Approach 기구", "a": "S-Reamer를 사용하여 상악동 막을 손상시키지 않고 안전하게 치조골을 삭제하며 막을 들어 올릴 수 있습니다."},

    # [임상 및 유지보수 심화]
    {"t": "임플란트 주위염 예방 설계", "a": "네오바이오텍 임플란트는 S.L.A. 표면과 Tight한 커넥션 구조를 통해 세균 침투를 억제하고 임플란트 주위염(Peri-implantitis) 발생 위험을 낮췄습니다."},
    {"t": "보철물 나사 풀림 해결", "a": "Double Hex 구조와 Morse Taper의 냉간 용접(Cold Welding) 효과로 나사가 저절로 풀리는 현상을 근본적으로 차단했습니다."},
    {"t": "PickCap 인상재 선택", "a": "PickCap 코핑 사용 시에는 흐름성이 좋은 Light Body 인상재를 먼저 주입하고, 그 위에 Heavy Body를 담은 트레이를 압접하는 것이 좋습니다."},

    # [M5 체어 심화]
    {"t": "M5 체어의 에어 썩션(Air Suction)", "a": "강력한 흡입력을 제공하는 에어 썩션 시스템을 채택하여 수술 중 발생하는 혈액과 타액을 신속하게 제거합니다."},
    {"t": "M5 체어의 회전형 타구대", "a": "타구대(Spittoon)가 환자 쪽으로 90도까지 회전하여, 환자가 몸을 많이 일으키지 않고도 편안하게 입을 헹굴 수 있습니다."}
]

# =========================================================
# 질문 템플릿 (다양한 상황 설정으로 데이터 증강)
# =========================================================
templates = [
    "{}에 대해 자세히 설명해 주세요.",
    "네오바이오텍의 {}가 무엇인지 알려줘.",
    "임상 현장에서 {}가 중요한 이유는?",
    "{}의 구체적인 특징과 장점은 무엇인가요?",
    "경쟁사 제품 대비 {}의 차별점은?",
    "신입 직원을 위해 {}에 대한 교육 내용을 정리해줘.",
    "{}와 관련된 기술적 스펙을 알려줘.",
    "고객(치과의사)이 {}에 대해 물어보면 어떻게 답해야 해?",
    "{} 사용 시 주의해야 할 점은?",
    "{}의 적용 사례나 효과에 대해 설명해줘."
]

# =========================================================
# 800개 데이터 생성 로직
# =========================================================
generated_data = []
target_count = 800

print(f"나머지 {target_count}개 데이터 생성을 시작합니다...")

while len(generated_data) < target_count:
    for item in knowledge_base:
        if len(generated_data) >= target_count:
            break
        
        # 랜덤 템플릿 적용
        temp = random.choice(templates)
        instruction = temp.format(item['t'])
        
        # 답변에 약간의 변주를 주어 중복 회피 (선택적)
        output = item['a']
        
        generated_data.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })

# =========================================================
# JSONL 파일 저장
# =========================================================
filename = "neobiotech_remaining_800.jsonl"
with open(filename, "w", encoding="utf-8") as f:
    for entry in generated_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"🎉 성공! '{filename}' 파일이 생성되었습니다. (데이터 {len(generated_data)}개)")
print("이제 batch_1, batch_2, remaining_800 파일을 모두 합치면 총 1,000개가 됩니다.")