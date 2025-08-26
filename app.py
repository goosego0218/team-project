import os, re, torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== 사용자 로컬 모델 경로 ======
# Windows 경로면 r"..."(raw string)로 적어 주세요.
LOCAL_MODEL_PATH = r"C:\자연어처리 A반\미니프로젝트\8월\kogpt2-itqg-best"

# ====== 특수 토큰: 질문/요약 종료 제어 ======
END_Q_TOKEN = "[END_Q]"
END_SUM_TOKEN = "[END_SUM]"
END_Q_ID = None
END_SUM_ID = None


def ensure_special_tokens(model, tokenizer):
    """pad_token 보정 + [END_Q], [END_SUM] 추가 후 임베딩 리사이즈"""
    global END_Q_ID, END_SUM_ID
    to_add = []

    # pad 토큰 보정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 종료 토큰 추가
    if END_Q_TOKEN not in tokenizer.get_vocab():
        to_add.append(END_Q_TOKEN)
    if END_SUM_TOKEN not in tokenizer.get_vocab():
        to_add.append(END_SUM_TOKEN)

    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        model.resize_token_embeddings(len(tokenizer))

    END_Q_ID = tokenizer.convert_tokens_to_ids(END_Q_TOKEN)
    END_SUM_ID = tokenizer.convert_tokens_to_ids(END_SUM_TOKEN)


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


# ====== 공통 유틸 ======
_KR_QUESTION_CUES = (
    "무엇", "왜", "어째서", "어떻게", "언제", "어디", "어느", "어떤",
    "가능", "말씀해", "설명해", "얘기해", "듣고 싶", "알고 싶", "궁금"
)

def _split_sentences_kr(text: str):
    """한/영 혼합 문장 경계를 대략적으로 분할"""
    text = text.strip()
    if not text:
        return []
    pattern = r'(?<=\.)\s+|(?<=\?)\s+|(?<=!)\s+|(?<=…)\s+|(?<=\))\s+'
    sents = re.split(pattern, text)
    # 빈 문자열 제거
    return [s.strip() for s in sents if s and s.strip()]


def _remove_question_like_sentences(text: str, keep_min=2, keep_max=4):
    """
    요약 후보에서 질문 형태(물음표 포함/질문 단어 포함) 문장을 제거.
    최소 2~4문장 범위로 잘라 반환(부족하면 비질문 문장 위주로 채움).
    """
    sents = _split_sentences_kr(text)
    if not sents:
        return text

    def is_question_like(s: str) -> bool:
        s_ = s.strip()
        if not s_:
            return False
        if s_.endswith("?"):
            return True
        # 질문 단서 포함 여부
        return any(cue in s_ for cue in _KR_QUESTION_CUES)

    non_q = [s for s in sents if not is_question_like(s)]
    if not non_q:
        # 전부 질문 형태면 물음표 제거 후 앞 문장 몇 개만 사용
        cleaned = [s.rstrip("?!.…") for s in sents][:keep_max]
        return " ".join(cleaned).strip()

    # 2~4문장 범위로 제한
    picked = non_q[:max(keep_min, min(keep_max, len(non_q)))]
    return " ".join(picked).strip()


# ====== 요약 생성 ======
def summarize_resume(model, tokenizer, resume, position="미지정"):
    """모델 요약(2~4문장). 실패 시 문장 추출 Fallback."""
    if model is None or tokenizer is None:
        sents = _split_sentences_kr(resume)
        return " ".join(sents[:3])[:600] if sents else resume[:300]

    SUM_PROMPT = (
        "[TASK] 아래 자기소개서를 2~4문장으로 간결 요약하세요. "
        "오직 요약만 작성하고, 어떤 형태의 질문도 작성하지 마세요. "
        "기술스택·역할·수치 성과 중심으로 요약하십시오.\n"
        f"[POSITION] {position}\n"
        f"[자기소개서]\n{resume}\n\n[요약]\n"
    )
    try:
        inputs = tokenizer(SUM_PROMPT, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.4,           # 보수적으로 낮춤(질문 폭주 방지)
                top_p=0.9,
                max_new_tokens=220,
                no_repeat_ngram_size=3,
                eos_token_id=END_SUM_ID,   # 요약 종료 토큰에서 멈춤
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        # [요약] 이후만 취하고, 다른 섹션/마커를 제거
        if "[요약]" in text:
            text = text.split("[요약]")[-1]
        text = re.split(r'\[질문\]|\[자기\s*소개서\]|###|\[END_Q\]', text)[0].strip()

        # 요약 내 질문 형태 문장 제거(2~4문장 유지)
        text = _remove_question_like_sentences(text, keep_min=2, keep_max=4)

        # 혹시 남은 물음표만 끝에 남아있다면 정리
        if text.endswith("?"):
            text = text.rstrip("?!.…").strip()
        return text[:1000] or summarize_resume(None, None, resume)
    except Exception:
        sents = _split_sentences_kr(resume)
        return " ".join(sents[:3])[:600] if sents else resume[:300]


# ====== 질문 생성 ======
TASK_PROMPT_TEMPLATE = (
    "[TASK] 다음 자기소개서를 읽고, 자기소개서의 내용을 기반으로, IT 채용 면접관 입장에서 "
    "핵심 역량·프로젝트에 대해 꼬리질문할 수 있는 구체적이고 날카로운 질문 1개만 작성하시오. "
    "반드시 한 문장, 한국어로 작성. 응답 끝에 [END_Q]를 붙이시오.\n"
    "[POSITION] {position}\n"
    "[자기소개서]\n{resume}\n\n"
    "[질문]\n"
)

def _cleanup_question(txt: str) -> str:
    t = " ".join(txt.strip().split())
    t = t.split(END_Q_TOKEN)[0]               # [END_Q]로 자르기
    t = re.split(r'\[자기\s*소개서\]|\[질의.*\]|###|\[요약\]', t)[0].strip()
    # 요약/서술형 문장 제거용 간단한 휴리스틱
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
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=64,
                num_return_sequences=max(1, min(n, 10)),
                no_repeat_ngram_size=3,
                eos_token_id=END_Q_ID,                 # [END_Q]에서 종료
                pad_token_id=tokenizer.eos_token_id,
            )

        qs = []
        for o in outputs:
            text = tokenizer.decode(o, skip_special_tokens=True)
            piece = text.split("[질문]")[-1] if "[질문]" in text else text
            q = _cleanup_question(piece)
            if q and q not in qs:
                qs.append(q)

        # 혹시나 문장 말미가 요약처럼 끝났다면 필터링
        qs = [q if q.endswith("?") else (q.rstrip(" .!…") + "?") for q in qs]
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


@app.route("/health", methods=["GET"])
def health():
    ok = model is not None and tokenizer is not None and END_Q_ID is not None and END_SUM_ID is not None
    return jsonify({"ok": ok})


if __name__ == "__main__":
    # 외부 접근 필요시 host="0.0.0.0"
    app.run(host="127.0.0.1", port=5000, debug=True)
