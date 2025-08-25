// components/MainContent.js
import React, { useState } from 'react';
import './MainContent.css';

/* ===== 검증 유틸: 제출 시에만 사용 ===== */
const URL_RE = /(https?:\/\/|www\.)/i;
const LETTER_RE = /[A-Za-z가-힣]/;
const VALID_JOB_CHARS_RE = /^[A-Za-z가-힣\s/&()\-·.]+$/; // 허용 문자셋
const REPEAT_CHAR_RE = /(.)\1{4,}/; // 동일문자 5회 이상 반복

function validateJobTitle(value) {
  const v = (value || '').trim();
  if (v.length < 2) return { ok: false, msg: '지원 직무는 최소 2자 이상이어야 합니다.' };
  if (v.length > 40) return { ok: false, msg: '지원 직무는 최대 40자까지 가능합니다.' };
  if (!LETTER_RE.test(v)) return { ok: false, msg: '지원 직무에는 한글/영문이 최소 1자 이상 포함되어야 합니다.' };
  if (!VALID_JOB_CHARS_RE.test(v)) return { ok: false, msg: '지원 직무에는 특수문자를 사용할 수 없습니다. (허용: 공백, / & ( ) - · .)' };
  if (URL_RE.test(v)) return { ok: false, msg: 'URL(링크) 형태의 입력은 허용되지 않습니다.' };
  if (REPEAT_CHAR_RE.test(v)) return { ok: false, msg: '동일 문자를 과도하게 반복한 입력은 허용되지 않습니다.' };
  return { ok: true };
}

function ratio(count, total) {
  if (!total) return 0;
  return count / total;
}

function validateCoverLetter(text) {
  const t = (text || '').trim();
  if (t.length < 200) return { ok: false, msg: '자기소개서는 최소 200자 이상 입력해야 합니다.' };
  if (t.length > 8000) return { ok: false, msg: '자기소개서는 최대 8000자까지 가능합니다.' };

  const noSpaceLen = t.replace(/\s/g, '').length;
  const letterCount = (t.match(/[A-Za-z가-힣]/g) || []).length;
  const symbolCount = (t.match(/[^A-Za-z가-힣0-9\s\n\r.,!?'"()\-;:/]/g) || []).length;

  if (ratio(letterCount, noSpaceLen) < 0.5) {
    return { ok: false, msg: '문자(한글/영문) 비율이 너무 낮습니다. 의미 있는 문장을 입력해 주세요.' };
  }
  if (ratio(symbolCount, noSpaceLen) > 0.2) {
    return { ok: false, msg: '특수기호가 과도하게 포함되어 있습니다.' };
  }
  if (REPEAT_CHAR_RE.test(t)) {
    return { ok: false, msg: '동일 문자를 과도하게 반복한 입력은 허용되지 않습니다.' };
  }

  // 너무 단순한 어휘(고유 단어 수 체크)
  const words = t
    .toLowerCase()
    .replace(/[^a-z가-힣0-9\s]/gi, ' ')
    .split(/\s+/)
    .filter(Boolean);
  const uniqueWords = new Set(words);
  if (uniqueWords.size < 30) {
    return { ok: false, msg: '내용이 지나치게 단순합니다. 더 구체적으로 작성해 주세요.' };
  }
  return { ok: true };
}

const MainContent = ({ currentView = 'analyze', onAnalysis, analysisData, isLoading }) => {
  const [inputText, setInputText] = useState('');
  const [position, setPosition] = useState('');
  const [inputMethod, setInputMethod] = useState('text');

  // ===== 커스텀 알럿 상태 =====
  const [alertMsg, setAlertMsg] = useState('');
  const showAlert = (msg) => setAlertMsg(String(msg || ''));
  const closeAlert = () => setAlertMsg('');

  const showAnalyze = currentView === 'analyze';
  const showResults = currentView === 'results' && !!analysisData;

  /* ===== 제출(시도) 시에만 검증하여 알럿 ===== */
  const handleTextSubmit = () => {
    const jobCheck = validateJobTitle(position);
    if (!jobCheck.ok) {
      showAlert(jobCheck.msg);
      return;
    }
    const coverCheck = validateCoverLetter(inputText);
    if (!coverCheck.ok) {
      showAlert(coverCheck.msg);
      return;
    }
    onAnalysis?.({ type: 'text', content: inputText, position });
  };

  const handleFileUpload = (files) => {
    const file = files?.[0];
    if (!file) return;

    const jobCheck = validateJobTitle(position);
    if (!jobCheck.ok) {
      showAlert(jobCheck.msg);
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const fileText = String(e.target.result || '');
      const coverCheck = validateCoverLetter(fileText);
      if (!coverCheck.ok) {
        showAlert(coverCheck.msg);
        return;
      }
      onAnalysis?.({ type: 'file', content: fileText, filename: file.name, position });
    };
    // .docx/.pdf는 텍스트 추출 품질 이슈가 있을 수 있음(백엔드 추출로 개선 권장)
    reader.readAsText(file);
  };

  return (
    <div className="main-content">
      {showAnalyze && (
        <div className="input-section">
          <div className="input-tabs">
            <button
              className={`tab-btn ${inputMethod === 'text' ? 'active' : ''}`}
              onClick={() => setInputMethod('text')}
              type="button"
            >
              <span className="icon">T</span>텍스트 입력
            </button>
            <button
              className={`tab-btn ${inputMethod === 'file' ? 'active' : ''}`}
              onClick={() => setInputMethod('file')}
              type="button"
            >
              <span className="icon">F</span>파일 업로드
            </button>
          </div>

          <div className="position-row" style={{ margin: '12px 0' }}>
            <label style={{ marginRight: 8, fontWeight: 600 }}>지원 직무</label>
            <input
              type="text"
              placeholder="예: 백엔드 개발자"
              value={position}
              onChange={(e) => setPosition(e.target.value)} // 입력 중엔 검증 X
              className="position-input"
              style={{ flex: 1, padding: '8px 12px', borderRadius: 8 }}
              maxLength={40}
            />
          </div>

          <div className="input-area">
            {inputMethod === 'text' ? (
              <div className="text-input-container">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)} // 입력 중엔 검증 X
                  placeholder="자기소개서를 붙여넣어 주세요 (최소 200자)"
                  rows={8}
                  className="text-input"
                  minLength={200}
                  maxLength={8000}
                />
                <button
                  className="analyze-btn"
                  onClick={handleTextSubmit}
                  disabled={!inputText.trim() || isLoading}
                  type="button"
                >
                  {isLoading ? '분석 중...' : '분석 시작'}
                </button>
              </div>
            ) : (
              <div className="file-input-container">
                <div className="file-drop-zone">
                  <span className="icon large">F</span>
                  <p>파일을 드래그하여 업로드하거나 클릭하여 선택하세요</p>
                  <p className="file-types">지원 형식: .txt, .doc, .docx, .pdf</p>
                  <input
                    type="file"
                    accept=".txt,.doc,.docx,.pdf"
                    onChange={(e) => handleFileUpload(e.target.files)}
                    className="file-input"
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {showResults && (
        <div className="results-section">
          <div className="results-grid">
            {/* 1) 요약 */}
            <div className="result-card summary-card">
              <div className="card-header"><h3>자기소개서 요약</h3></div>
              <div className="card-content"><p>{analysisData.summary}</p></div>
            </div>

            {/* 2) 예상 면접 질문 */}
            <div className="result-card questions-card">
              <div className="card-header"><h3>예상 면접 질문</h3></div>
              <div className="card-content">
                <ul className="questions-list">
                  {analysisData.questions.map((q, i) => (
                    <li key={i} className="question-item">
                      <span className="icon">Q</span>
                      <span>{`${i + 1}. ${q}`}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 커스텀 알럿 모달 */}
      {alertMsg && (
        <div className="custom-alert-overlay" onClick={closeAlert}>
          <div
            className="custom-alert-box"
            role="alertdialog"
            aria-live="assertive"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="custom-alert-icon">!</div>
            <p className="custom-alert-message">{alertMsg}</p>
            <button className="custom-alert-button" onClick={closeAlert} autoFocus type="button">
              확인
            </button>
          </div>
        </div>
      )}

      {currentView === 'results' && !analysisData && (
        <div className="results-section">
          <div className="result-card">
            <div className="card-header"><h3>분석 결과가 없습니다</h3></div>
            <div className="card-content">
              <p>먼저 상단에서 자기소개서를 입력/업로드한 뒤 “분석 시작”을 눌러주세요.</p>
            </div>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>자기소개서를 분석하고 있습니다...</p>
        </div>
      )}
    </div>
  );
};

export default MainContent;
