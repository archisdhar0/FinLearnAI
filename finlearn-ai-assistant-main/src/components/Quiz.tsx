import { useState } from 'react';
import { CheckCircle, XCircle, RefreshCw, Trophy } from 'lucide-react';

interface QuizQuestion {
  question: string;
  options: string[];
  correct: number;
  explanation?: string;
}

interface QuizProps {
  questions: QuizQuestion[];
  onComplete: (score: number, total: number) => void;
  title?: string;
  allowRetake?: boolean;
}

export const Quiz = ({ questions, onComplete, title = "Quiz", allowRetake = true }: QuizProps) => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>(new Array(questions.length).fill(null));
  const [isComplete, setIsComplete] = useState(false);

  const handleAnswer = (index: number) => {
    if (showResult) return;
    setSelectedAnswer(index);
  };

  const submitAnswer = () => {
    if (selectedAnswer === null) return;
    
    const newAnswers = [...answers];
    newAnswers[currentQuestion] = selectedAnswer;
    setAnswers(newAnswers);
    
    if (selectedAnswer === questions[currentQuestion].correct) {
      setScore(score + 1);
    }
    setShowResult(true);
  };

  const nextQuestion = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setSelectedAnswer(null);
      setShowResult(false);
    } else {
      // Quiz complete
      const finalScore = answers.reduce((acc, ans, idx) => {
        return acc + (ans === questions[idx].correct ? 1 : 0);
      }, selectedAnswer === questions[currentQuestion].correct ? 1 : 0);
      
      setIsComplete(true);
      onComplete(finalScore, questions.length);
    }
  };

  const retakeQuiz = () => {
    setCurrentQuestion(0);
    setSelectedAnswer(null);
    setShowResult(false);
    setScore(0);
    setAnswers(new Array(questions.length).fill(null));
    setIsComplete(false);
  };

  if (isComplete) {
    const percentage = Math.round((score / questions.length) * 100);
    const passed = percentage >= 70;
    
    return (
      <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
        <div className="text-center">
          <div className={`inline-flex items-center justify-center w-20 h-20 rounded-full mb-4 ${
            passed ? 'bg-green-500/20' : 'bg-yellow-500/20'
          }`}>
            <Trophy className={`w-10 h-10 ${passed ? 'text-green-400' : 'text-yellow-400'}`} />
          </div>
          
          <h3 className="text-2xl font-bold text-white mb-2">
            {passed ? 'Great Job!' : 'Keep Learning!'}
          </h3>
          
          <p className="text-4xl font-bold mb-2">
            <span className={passed ? 'text-green-400' : 'text-yellow-400'}>{score}</span>
            <span className="text-slate-400">/{questions.length}</span>
          </p>
          
          <p className="text-slate-400 mb-6">
            You scored {percentage}% {passed ? '- Quiz Passed!' : '- 70% needed to pass'}
          </p>
          
          {allowRetake && !passed && (
            <button
              onClick={retakeQuiz}
              className="flex items-center gap-2 mx-auto px-6 py-3 bg-primary text-white rounded-lg hover:opacity-90 transition-opacity"
            >
              <RefreshCw className="w-4 h-4" />
              Retake Quiz
            </button>
          )}
        </div>
      </div>
    );
  }

  const question = questions[currentQuestion];
  const isCorrect = selectedAnswer === question.correct;

  return (
    <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
        <span className="text-sm text-slate-400">
          Question {currentQuestion + 1} of {questions.length}
        </span>
      </div>
      
      {/* Progress bar */}
      <div className="h-1 bg-slate-700 rounded-full mb-6">
        <div 
          className="h-1 bg-primary rounded-full transition-all"
          style={{ width: `${((currentQuestion + (showResult ? 1 : 0)) / questions.length) * 100}%` }}
        />
      </div>
      
      <p className="text-white text-lg mb-6">{question.question}</p>
      
      <div className="space-y-3 mb-6">
        {question.options.map((option, index) => {
          let bgColor = 'bg-slate-900/50 hover:bg-slate-900';
          let borderColor = 'border-slate-700';
          
          if (showResult) {
            if (index === question.correct) {
              bgColor = 'bg-green-500/20';
              borderColor = 'border-green-500';
            } else if (index === selectedAnswer && index !== question.correct) {
              bgColor = 'bg-red-500/20';
              borderColor = 'border-red-500';
            }
          } else if (selectedAnswer === index) {
            borderColor = 'border-primary';
            bgColor = 'bg-primary/10';
          }
          
          return (
            <button
              key={index}
              onClick={() => handleAnswer(index)}
              disabled={showResult}
              className={`w-full p-4 rounded-lg border ${borderColor} ${bgColor} text-left transition-all flex items-center gap-3`}
            >
              <span className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-sm font-medium">
                {String.fromCharCode(65 + index)}
              </span>
              <span className="text-white flex-1">{option}</span>
              {showResult && index === question.correct && (
                <CheckCircle className="w-5 h-5 text-green-400" />
              )}
              {showResult && index === selectedAnswer && index !== question.correct && (
                <XCircle className="w-5 h-5 text-red-400" />
              )}
            </button>
          );
        })}
      </div>
      
      {showResult && question.explanation && (
        <div className={`p-4 rounded-lg mb-6 ${isCorrect ? 'bg-green-500/10 border border-green-500/30' : 'bg-yellow-500/10 border border-yellow-500/30'}`}>
          <p className="text-sm text-slate-300">
            <strong className={isCorrect ? 'text-green-400' : 'text-yellow-400'}>
              {isCorrect ? 'Correct! ' : 'Explanation: '}
            </strong>
            {question.explanation}
          </p>
        </div>
      )}
      
      <div className="flex justify-end">
        {!showResult ? (
          <button
            onClick={submitAnswer}
            disabled={selectedAnswer === null}
            className="px-6 py-2 bg-primary text-white rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
          >
            Check Answer
          </button>
        ) : (
          <button
            onClick={nextQuestion}
            className="px-6 py-2 bg-primary text-white rounded-lg hover:opacity-90 transition-opacity"
          >
            {currentQuestion === questions.length - 1 ? 'See Results' : 'Next Question'}
          </button>
        )}
      </div>
    </div>
  );
};

export default Quiz;
