# app.py
# pip install flask flask-cors transformers torch flask-swagger-ui

import os, re, torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== 사용자 로컬 모델 경로 ======
# Windows 경로면 r"..."(raw string)로 적어 주세요.
LOCAL_MODEL_PATH = r"C:\자연어처리 A반\미니프로젝트\8월\kogpt2-itqg-best"

# ====== 특수 토큰([END_Q])로 생성 종료 ======
END_Q_TOKEN = "[END_Q]"
END_ID = None

def ensure_special_tokens(model, tokenizer):
    """pad_token 보정 + [END_Q] 추가 후 임베딩 리사이즈"""
    global END_ID
    added = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        added = True

    if END_Q_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [END_Q_TOKEN]})
        added = True

    END_ID = tokenizer.convert_tokens_to_ids(END_Q_TOKEN)

    if added:
        model.resize_token_embeddings(len(tokenizer))

def load_model_locally(model_path):
    """로컬 저장 모델 로드"""
    print(f"[LOAD] {model_path}")
    if not os.path.exists(model_path):
        print("[ERROR] 모델 경로가 없습니다.")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map)
        model.eval()

        ensure_special_tokens(model, tokenizer)
        print("[OK] 모델 로딩 완료")
        return model, tokenizer
    except Exception as e:
        print("[ERROR] 모델 로딩 실패:", e)
        return None, None

# ====== Flask ======
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model, tokenizer = load_model_locally(LOCAL_MODEL_PATH)

# ====== 요약 생성 ======
def summarize_resume(model, tokenizer, resume, position="미지정"):
    """모델 요약(2~4문장). 실패 시 문장 추출 Fallback."""
    if model is None or tokenizer is None:
        sents = re.split(r'(?<=[.!?]|[가-힣]\))\s+', resume.strip())
        return ' '.join(sents[:3])[:600] if sents else resume[:300]

    SUM_PROMPT = (
        "[TASK] 아래 자기소개서를 2~4문장으로 간결 요약하세요. "
        "기술스택·역할·수치 성과 중심으로.\n"
        f"[POSITION] {position}\n"
        f"[자기소개서]\n{resume}\n\n[요약]\n"
    )
    try:
        inputs = tokenizer(SUM_PROMPT, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.5,    # 노트북과 유사 샘플링
                top_p=0.9,
                max_new_tokens=180,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if "[요약]" in text:
            text = text.split("[요약]")[-1].strip()
        text = re.split(r'\[질문\]|\[자기소개서\]|###', text)[0].strip()
        return text[:1000] or summarize_resume(None, None, resume)
    except Exception:
        sents = re.split(r'(?<=[.!?]|[가-힣]\))\s+', resume.strip())
        return ' '.join(sents[:3])[:600] if sents else resume[:300]

# ====== 질문 생성 ======
TASK_PROMPT_TEMPLATE = (
    "[TASK] 다음 자기소개서를 읽고, 자기소개서의 내용을 기반으로, IT 채용 면접관 입장에서 "
    "핵심 역량·프로젝트에 대해 꼬리질문할 수 있는 구체적이고 날카로운 질문 1개만 작성하시오. "
    "반드시 한 문장, 한국어로 작성.\n"
    "[POSITION] {position}\n"
    "[자기소개서]\n{resume}\n\n"
    "[질문]\n"
)

def _cleanup_question(txt: str) -> str:
    t = " ".join(txt.strip().split())
    t = t.split(END_Q_TOKEN)[0]              # [END_Q]로 자르기
    t = re.split(r'\[자기\s*소개서\]|\[질의.*\]|###', t)[0].strip()
    if not t.endswith("?"):
        t = t.rstrip(" .!…") + "?"
    return t

def generate_questions(model, tokenizer, position, resume, n=5):
    if model is None or tokenizer is None:
        return ["모델이 로드되지 않았습니다."]
    prompt = TASK_PROMPT_TEMPLATE.format(position=position.strip(), resume=resume.strip())

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,    # 노트북값
                top_p=0.9,
                max_new_tokens=50,
                num_return_sequences=max(1, min(n, 10)),
                no_repeat_ngram_size=3,
                eos_token_id=END_ID,                    # [END_Q]에서 종료
                pad_token_id=tokenizer.eos_token_id,
            )

        qs = []
        for o in outputs:
            text = tokenizer.decode(o, skip_special_tokens=True)
            piece = text.split("[질문]")[-1] if "[질문]" in text else text
            q = _cleanup_question(piece)
            if q and q not in qs:
                qs.append(q)
        return qs or ["질문 생성 실패. 입력을 다시 시도해 주세요."]
    except Exception as e:
        print("[ERROR] 질문 생성:", e)
        return ["질문 생성 중 오류가 발생했습니다."]

# ====== API ======
@app.route("/generate", methods=["POST"])
def generate():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(force=True, silent=True) or {}
    resume = (data.get("resume") or "").strip()
    position = data.get("position", "미지정")
    num_questions = int(data.get("num_questions", 5))

    if not resume:
        return jsonify({"error": "resume is required"}), 400

    summary = summarize_resume(model, tokenizer, resume, position)
    questions = generate_questions(model, tokenizer, position, resume, n=num_questions)

    return jsonify({"summary": summary, "questions": questions})

if __name__ == "__main__":
    # 외부 접근 필요시 host="0.0.0.0"
    app.run(host="127.0.0.1", port=5000, debug=True)