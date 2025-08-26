import os, re, json, glob, difflib, math, torch
from collections import Counter
from typing import Iterable, Tuple, List, Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 설정
# =========================
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", r"C:\자연어처리 A반\미니프로젝트\8월\kogpt2-itqg-best")

# JSONL 경로 지정(선택). 비우면 자동 탐색(**/*prompt_completion.jsonl)
TRAIN_JSONL = r"C:\자연어처리 A반\미니프로젝트\8월\train_prompt_completion.jsonl"
VAL_JSONL   = r"C:\자연어처리 A반\미니프로젝트\8월\validation_prompt_completion.jsonl"
TEST_JSONL  = r"C:\자연어처리 A반\미니프로젝트\8월\test_prompt_completion.jsonl"

END_Q_TOKEN = "[END_Q]"
END_ID = None

# =========================
# 정규식/유틸
# =========================
TECH_RE   = re.compile(r"[A-Za-z][A-Za-z0-9+\-\.#]{2,}")
KR_WORD_RE= re.compile(r"[가-힣]{2,}")

def log(msg: str):  # 깔끔한 로그
    print(msg, flush=True)

def parse_resume_from_prompt(prompt: str) -> str:
    """노트북 포맷: [자기소개서] ... [질문] 사이를 추출"""
    m = re.search(r"\[자기소개서\]\s*(.*?)\s*\n\s*\[질문\]", prompt, flags=re.S)
    return m.group(1).strip() if m else ""

def extract_anchors(text: str, topk: int = 6):
    toks = TECH_RE.findall(text)
    if not toks: return []
    freq = Counter([t.strip(".") for t in toks])
    return [w for w, _ in freq.most_common(topk)]

def post_cleanup_kr(q: str) -> str:
    q = q.strip()
    q = re.sub(r"(할|했을|하는)\s*,\s*", r"\1 때, ", q)
    q = re.sub(r"\b([가-힣]{2,})\s*\1\b", r"\1", q)
    q = re.sub(r"\s{2,}", " ", q)
    q = re.sub(r"\s+,", ",", q)
    q = re.sub(r",\s*", ", ", q)
    q = re.sub(r"\s+\?", "?", q)
    if not q.endswith("?"):
        q = q.rstrip(".!…") + "?"
    return q.strip()

def autocorrect_tech_terms(q: str, lexicon: set):
    canon = {w.lower(): w for w in lexicon}
    keys = list(canon.keys())
    parts = re.findall(r"\w+|\W+", q, flags=re.UNICODE)
    out = []
    for p in parts:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9+\-\.#]{2,}", p):
            k = p.lower()
            if k not in canon and len(p) >= 4 and keys:
                cand = difflib.get_close_matches(k, keys, n=1, cutoff=0.82)
                if cand:
                    p = canon[cand[0]]
        out.append(p)
    return "".join(out)

def autocorrect_korean_terms(q: str, lexicon: set, freq: dict, cutoff=0.90):
    if not lexicon: return q
    parts = re.findall(r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9+\-\.#]{2,}|\s+|[^\w\s]", q, flags=re.UNICODE)
    out = []
    for p in parts:
        if p.isspace() or re.fullmatch(r"[^\w\s]", p):
            out.append(p); continue
        if KR_WORD_RE.fullmatch(p) and p not in lexicon and len(p) >= 2:
            cands = difflib.get_close_matches(p, list(lexicon), n=3, cutoff=cutoff)
            if cands:
                p = max(cands, key=lambda w: freq.get(w, 0))
        out.append(p)
    return "".join(out)

def lexicon_coverage_score(q: str, kr_lexicon: set, anchors=None):
    if not q: return 0.0
    words = KR_WORD_RE.findall(q)
    cov = (sum(1 for w in words if w in kr_lexicon) / max(1, len(words)))
    hit = 0.0
    if anchors:
        lower = q.lower()
        hit = sum(1 for a in anchors if a.lower() in lower) / max(1, len(anchors))
    return 0.6 * cov + 0.4 * hit

def _iter_jsonl_rows(path: str):
    if not os.path.exists(path):
        return
    # BOM/인코딩 안전 처리
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
            return
        except Exception:
            continue

def _extract_from_row(row: Dict[str, Any]) -> Tuple[str, str]:
    """여러 스키마 케이스를 지원."""
    # 1) 표준 케이스
    prompt = str(row.get("prompt", "") or "")
    completion = row.get("completion", "")

    # 2) 다른 필드명 케이스
    resume_alt = str(row.get("resume", "") or "")
    q_alt = row.get("interview_question", "")

    # 3) completion이 dict인 케이스
    if isinstance(completion, dict):
        # {"question":"..."} 또는 {"text":"..."} 등
        completion = completion.get("question") or completion.get("text") or ""

    # resume 추출
    resume_text = ""
    if prompt:
        resume_text = parse_resume_from_prompt(prompt)
    if not resume_text and resume_alt:
        resume_text = resume_alt.strip()

    # 질문 추출
    question_text = ""
    if isinstance(completion, str) and completion:
        question_text = completion.replace(END_Q_TOKEN, "").strip()
    if not question_text and isinstance(q_alt, str):
        question_text = q_alt.strip()

    return resume_text, question_text

def _collect_texts_from_jsonls(paths: Iterable[str]) -> Tuple[List[str], List[str], Dict[str, int]]:
    resumes, questions = [], []
    file_read_stats = {}
    for p in paths:
        cnt = 0
        for row in _iter_jsonl_rows(p):
            res, q = _extract_from_row(row)
            if res: resumes.append(res); cnt += 1
            if q:   questions.append(q)
        file_read_stats[p] = cnt
    return resumes, questions, file_read_stats

def build_global_en_lexicon_from_texts(texts, min_freq=3, topk=5000):
    toks = []
    for t in texts:
        toks += TECH_RE.findall(t)
    freq = Counter([w.strip(".") for w in toks])
    kept = [w for w, n in freq.most_common(topk) if n >= min_freq]
    return set(kept)

def build_kr_lexicon_from_texts(texts, min_freq=3, topk=20000):
    tokens = []
    for t in texts:
        tokens += KR_WORD_RE.findall(t)
    freq = Counter(tokens)
    kept = {w: n for w, n in freq.most_common(topk) if n >= min_freq}
    return set(kept.keys()), kept

def find_jsonl_candidates() -> List[str]:
    # 명시적 경로 우선
    explicit = [p for p in (TRAIN_JSONL, VAL_JSONL, TEST_JSONL) if p]
    explicit = [os.path.abspath(p) for p in explicit if os.path.exists(p)]
    # 자동 탐색
    auto = [os.path.abspath(p) for p in glob.glob("**/*prompt_completion.jsonl", recursive=True)]
    # 중복 제거, 존재하는 것만
    seen, paths = set(), []
    for p in explicit + auto:
        if os.path.exists(p) and p not in seen:
            seen.add(p); paths.append(p)
    return paths

# =========================
# 모델/토크나이저
# =========================
def ensure_special_tokens(model, tokenizer):
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
    log(f"[LOAD] {model_path}")
    if not os.path.exists(model_path):
        log("[ERROR] 모델 경로가 없습니다.")
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device_map = "auto" if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map)
        model.eval()
        ensure_special_tokens(model, tokenizer)
        log("[OK] 모델 로딩 완료")
        return model, tokenizer
    except Exception as e:
        log(f"[ERROR] 모델 로딩 실패: {e}")
        return None, None

# =========================
# Flask
# =========================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model, tokenizer = load_model_locally(LOCAL_MODEL_PATH)

GLOBAL_EN_LEXICON: set = set()
KR_LEXICON: set = set()
KR_FREQ: Dict[str, int] = {}

def load_lexicons(verbose=True):
    global GLOBAL_EN_LEXICON, KR_LEXICON, KR_FREQ
    paths = find_jsonl_candidates()
    if verbose:
        if paths:
            log("[LEXICON] 탐색된 JSONL:")
            for p in paths: log(f"  - {p}")
        else:
            log("[LEXICON] JSONL 파일을 찾지 못했습니다. 실행 폴더/경로를 확인하세요.")
    resumes, questions, stats = _collect_texts_from_jsonls(paths)
    if verbose and stats:
        for p, n in stats.items():
            log(f"[LEXICON] {os.path.basename(p)}: parsed resumes={n}, questions≈{n}")
    # 사전 생성
    GLOBAL_EN_LEXICON = build_global_en_lexicon_from_texts(resumes + questions, min_freq=3, topk=5000)
    KR_LEXICON, KR_FREQ = build_kr_lexicon_from_texts(resumes + questions, min_freq=3, topk=20000)
    log(f"[LEXICON] EN={len(GLOBAL_EN_LEXICON)}, KR={len(KR_LEXICON)}")
    if len(GLOBAL_EN_LEXICON) == 0 and len(KR_LEXICON) == 0:
        log("[WARN] 사전 크기가 0입니다. 경로/스키마를 확인하세요. (서버는 동작합니다)")

load_lexicons(verbose=True)

# =========================
# 요약/질문 생성
# =========================
TASK_PROMPT_TEMPLATE = (
    "[TASK] 다음 자기소개서를 읽고, 자기소개서의 내용을 기반으로, IT 채용 면접관 입장에서 "
    "핵심 역량·프로젝트에 대해 꼬리질문할 수 있는 구체적이고 날카로운 질문 1개만 작성하시오. "
    "반드시 한 문장, 한국어로 작성.\n"
    "[POSITION] {position}\n"
    "[자기소개서]\n{resume}\n\n"
    "[질문]\n"
)

def summarize_resume(model, tokenizer, resume, position="미지정"):
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
                **inputs, do_sample=True, temperature=0.5, top_p=0.9,
                max_new_tokens=180, no_repeat_ngram_size=3,
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

def _cleanup_question_raw(txt: str) -> str:
    t = " ".join(txt.strip().split())
    t = t.split(END_Q_TOKEN)[0]
    t = re.split(r'\[자기\s*소개서\]|\[질의.*\]|###', t)[0].strip()
    if not t.endswith("?"):
        t = t.rstrip(" .!…") + "?"
    return t

def _postprocess_and_score(q_raw: str, resume_text: str):
    anchors = extract_anchors(resume_text, topk=6)
    tech_lexicon = set(anchors) | GLOBAL_EN_LEXICON
    q = autocorrect_tech_terms(q_raw, tech_lexicon)
    q = autocorrect_korean_terms(q, KR_LEXICON, KR_FREQ, cutoff=0.90)
    q = post_cleanup_kr(q)
    score = lexicon_coverage_score(q, KR_LEXICON, anchors)
    return q, score

def generate_questions(model, tokenizer, position, resume, n=5):
    if model is None or tokenizer is None:
        return ["모델이 로드되지 않았습니다."]
    prompt = TASK_PROMPT_TEMPLATE.format(position=position.strip(), resume=resume.strip())
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True, temperature=0.7, top_p=0.9,
                max_new_tokens=50, num_return_sequences=max(1, min(n, 10)),
                no_repeat_ngram_size=3,
                eos_token_id=END_ID,
                pad_token_id=tokenizer.eos_token_id,
            )
        cand = []
        for o in outputs:
            text = tokenizer.decode(o, skip_special_tokens=True)
            piece = text.split("[질문]")[-1] if "[질문]" in text else text
            q_raw = _cleanup_question_raw(piece)
            q_fixed, score = _postprocess_and_score(q_raw, resume)
            cand.append((score, q_fixed))
        # 중복 제거 + 리랭킹
        uniq, seen = [], set()
        for s, q in sorted(cand, key=lambda x: x[0], reverse=True):
            if q not in seen:
                seen.add(q); uniq.append(q)
            if len(uniq) >= n: break
        return uniq or ["질문 생성 실패. 입력을 다시 시도해 주세요."]
    except Exception as e:
        log(f"[ERROR] 질문 생성: {e}")
        return ["질문 생성 중 오류가 발생했습니다."]

# =========================
# API
# =========================
@app.route("/generate", methods=["POST"])
def api_generate():
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

@app.route("/reload_lexicon", methods=["POST"])
def api_reload_lexicon():
    load_lexicons(verbose=True)
    return jsonify({"status": "ok", "en_terms": len(GLOBAL_EN_LEXICON), "kr_terms": len(KR_LEXICON)})

@app.route("/debug_lexicon", methods=["GET"])
def api_debug_lexicon():
    paths = find_jsonl_candidates()
    resumes, questions, stats = _collect_texts_from_jsonls(paths)
    return jsonify({
        "cwd": os.getcwd(),
        "found_paths": paths,
        "file_read_stats": stats,
        "resume_samples": resumes[:2],
        "question_samples": questions[:2],
        "en_terms": len(GLOBAL_EN_LEXICON),
        "kr_terms": len(KR_LEXICON),
    })

if __name__ == "__main__":
    # 외부 접속 시 host="0.0.0.0"
    app.run(host="127.0.0.1", port=5000, debug=True)
