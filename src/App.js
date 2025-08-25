// App.js
import React, { useState, useEffect } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import MainContent from './components/MainContent';

function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [currentView, setCurrentView] = useState('analyze');

  useEffect(() => {
    document.body.classList.add('theme-light');
    return () => document.body.classList.remove('theme-light');
  }, []);

  const handleAnalysis = async (payload) => {
    setIsLoading(true);
    setAnalysisData(null);

    const API_BASE = process.env.REACT_APP_API_BASE || 'http://127.0.0.1:5000';
    const API_URL = `${API_BASE}/generate`;

    const resume =
      payload?.type === 'file' ? payload.content : (payload?.content || '');
    const position = payload?.position || '미지정';

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resume, position, num_questions: 5 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text()}`);
      const result = await res.json();

      setAnalysisData({
        summary: result?.summary || '요약을 생성하지 못했습니다.',
        questions: result?.questions || [],
      });
      setCurrentView('results');
    } catch (e) {
      console.error(e);
      setAnalysisData({
        summary: '오류 발생',
        questions: [`질문 생성 오류: ${e.message}`],
      });
      setCurrentView('results');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`app ${isSidebarOpen ? '' : 'sidebar-collapsed'}`}>
      <Sidebar isOpen={isSidebarOpen} currentView={currentView} onSelectView={setCurrentView} />
      <div className="main-container">
        <Header onToggleSidebar={() => setIsSidebarOpen((v) => !v)} isSidebarOpen={isSidebarOpen} />
        <MainContent
          currentView={currentView}
          onAnalysis={handleAnalysis}
          analysisData={analysisData}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}
export default App;
