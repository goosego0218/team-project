import React, { useState, useEffect } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import MainContent from './components/MainContent';

function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [currentView, setCurrentView] = useState('analyze'); // 'analyze' | 'results'

  useEffect(() => {
    document.body.classList.add('theme-light');
    return () => {
      document.body.classList.remove('theme-light');
    };
  }, []);

  const handleAnalysis = async (data) => {
    setIsLoading(true);
    try {
      // 백엔드 API 호출 시뮬레이션 (백엔드 엔드포인트로 변경필요)
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // 임시 데이터 (백엔드에서 받을 데이터)
      const mockData = {
        summary: "지원자는 IT 분야에서 5년간의 경험을 가지고 있으며, 웹 개발과 데이터 분석에 특화되어 있습니다. 팀워크와 문제 해결 능력이 뛰어나며, 지속적인 학습을 통해 새로운 기술을 습득하는 것에 관심이 많습니다.",
        questions: [
          "웹 개발 프로젝트에서 가장 어려웠던 기술적 도전은 무엇이었나요?",
          "데이터 분석 경험 중 비즈니스 성과에 직접적으로 기여한 사례가 있나요?",
          "새로운 기술을 학습할 때 어떤 방법을 사용하시나요?",
          "팀 프로젝트에서 갈등이 발생했을 때 어떻게 해결하셨나요?"
        ],
        improvements: [
          "구체적인 성과 수치를 포함하여 경험을 설명하면 더욱 설득력 있을 것 같습니다.",
          "기술적 문제 해결 과정을 단계별로 설명하면 좋겠습니다.",
          "팀워크 경험에서 본인의 역할과 기여도를 구체적으로 언급하면 좋겠습니다."
        ]
      };
      
      setAnalysisData(mockData);
      setCurrentView('results');
    } catch (error) {
      console.error('분석 중 오류가 발생했습니다:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`app ${isSidebarOpen ? '' : 'sidebar-collapsed'}`}>
      <Sidebar 
        isOpen={isSidebarOpen}
        currentView={currentView}
        onSelectView={setCurrentView}
      />
      <div className="main-container">
        <Header 
          onToggleSidebar={() => setIsSidebarOpen(prev => !prev)}
          isSidebarOpen={isSidebarOpen}
        />
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
