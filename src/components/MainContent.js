import React, { useState } from 'react';
import './MainContent.css';

const MainContent = ({ currentView = 'analyze', onAnalysis, analysisData, isLoading }) => {
  const [inputText, setInputText] = useState('');
  const [inputMethod, setInputMethod] = useState('text'); // 'text' or 'file'

  const handleTextSubmit = () => {
    if (inputText.trim()) {
      onAnalysis({ type: 'text', content: inputText });
    }
  };

  const handleFileUpload = (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target.result;
        onAnalysis({ type: 'file', content, filename: file.name });
      };
      reader.readAsText(file);
    }
  };

  const showAnalyze = currentView === 'analyze';
  const showResults = currentView === 'results' && !!analysisData;

  return (
    <div className="main-content">
      {/* 입력 섹션 */}
      {showAnalyze && (
      <div className="input-section">
        <div className="input-tabs">
          <button 
            className={`tab-btn ${inputMethod === 'text' ? 'active' : ''}`}
            onClick={() => setInputMethod('text')}
          >
            <span className="icon">T</span>
            텍스트 입력
          </button>
          <button 
            className={`tab-btn ${inputMethod === 'file' ? 'active' : ''}`}
            onClick={() => setInputMethod('file')}
          >
            <span className="icon">F</span>
            파일 업로드
          </button>
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
              <button 
                className="analyze-btn"
                onClick={handleTextSubmit}
                disabled={!inputText.trim() || isLoading}
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

      {/* 분석 결과 섹션 */}
      {showResults && (
        <div className="results-section">
          <div className="results-grid">
            {/* 요약 결과 */}
            <div className="result-card summary-card">
              <div className="card-header">
                <h3>자기소개서 요약</h3>
              </div>
              <div className="card-content">
                <p>{analysisData.summary}</p>
              </div>
            </div>

            {/* 질문 생성 */}
            <div className="result-card questions-card">
              <div className="card-header">
                <h3>예상 면접 질문</h3>
              </div>
              <div className="card-content">
                <ul className="questions-list">
                  {analysisData.questions.map((question, index) => (
                    <li key={index} className="question-item">
                      <span className="icon">Q</span>
                      <span>{question}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* 보완사항 */}
            <div className="result-card improvements-card">
              <div className="card-header">
                <h3>개선 제안</h3>
              </div>
              <div className="card-content">
                <ul className="improvements-list">
                  {analysisData.improvements.map((improvement, index) => (
                    <li key={index} className="improvement-item">
                      <span className="icon">I</span>
                      <span>{improvement}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 결과 탭 선택 + 데이터 없음 상태 */}
      {currentView === 'results' && !analysisData && (
        <div className="results-section">
          <div className="result-card">
            <div className="card-header">
              <h3>분석 결과가 없습니다</h3>
            </div>
            <div className="card-content">
              <p>먼저 상단 입력 영역에서 자기소개서를 입력하거나 파일을 업로드한 뒤, "분석 시작"을 눌러 결과를 생성하세요.</p>
            </div>
          </div>
        </div>
      )}

      {/* 로딩 상태 */}
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
