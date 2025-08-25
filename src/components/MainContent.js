// components/MainContent.js
import React, { useState } from 'react';
import './MainContent.css';

const MainContent = ({ currentView = 'analyze', onAnalysis, analysisData, isLoading }) => {
  const [inputText, setInputText] = useState('');
  const [position, setPosition] = useState('');
  const [inputMethod, setInputMethod] = useState('text');

  const handleTextSubmit = () => {
    if (inputText.trim()) onAnalysis({ type: 'text', content: inputText, position });
  };

  const handleFileUpload = (files) => {
    const file = files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      onAnalysis({ type: 'file', content: e.target.result, filename: file.name, position });
    };
    // .docx/.pdf는 텍스트 추출 품질이 떨어질 수 있음(추후 백엔드 업로드로 확장 가능)
    reader.readAsText(file);
  };

  const showAnalyze = currentView === 'analyze';
  const showResults = currentView === 'results' && !!analysisData;

  return (
    <div className="main-content">
      {showAnalyze && (
        <div className="input-section">
          <div className="input-tabs">
            <button className={`tab-btn ${inputMethod === 'text' ? 'active' : ''}`} onClick={() => setInputMethod('text')}>
              <span className="icon">T</span>텍스트 입력
            </button>
            <button className={`tab-btn ${inputMethod === 'file' ? 'active' : ''}`} onClick={() => setInputMethod('file')}>
              <span className="icon">F</span>파일 업로드
            </button>
          </div>

          <div className="position-row" style={{ margin: '12px 0' }}>
            <label style={{ marginRight: 8, fontWeight: 600 }}>지원 직무</label>
            <input
              type="text"
              placeholder="예: 백엔드 개발자"
              value={position}
              onChange={(e) => setPosition(e.target.value)}
              className="position-input"
              style={{ flex: 1, padding: '8px 12px', borderRadius: 8 }}
            />
          </div>

          <div className="input-area">
            {inputMethod === 'text' ? (
              <div className="text-input-container">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="자기소개서 내용을 입력하세요..."
                  rows={8}
                  className="text-input"
                />
                <button className="analyze-btn" onClick={handleTextSubmit} disabled={!inputText.trim() || isLoading}>
                  {isLoading ? '분석 중...' : '분석 시작'}
                </button>
              </div>
            ) : (
              <div className="file-input-container">
                <div className="file-drop-zone">
                  <span className="icon large">F</span>
                  <p>파일을 드래그하여 업로드하거나 클릭하여 선택하세요</p>
                  <p className="file-types">지원 형식: .txt, .doc, .docx, .pdf</p>
                  <input type="file" accept=".txt,.doc,.docx,.pdf" onChange={(e) => handleFileUpload(e.target.files)} className="file-input" />
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
