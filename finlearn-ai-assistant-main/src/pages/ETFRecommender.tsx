import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import {
  ArrowLeft,
  ArrowRight,
  TrendingUp,
  Shield,
  Target,
  Zap,
  PieChart,
  Play,
  RefreshCw,
  Info,
  ChevronRight,
  DollarSign,
  Calendar,
  BarChart3,
} from "lucide-react";

// ETF data with historical returns and volatility
const ETF_DATA: Record<string, {
  name: string;
  category: string;
  expenseRatio: number;
  avgReturn: number;      // Historical average annual return
  volatility: number;     // Standard deviation of returns
  description: string;
}> = {
  // US Stocks
  VTI: { name: "Vanguard Total Stock Market", category: "US Stocks", expenseRatio: 0.03, avgReturn: 10.5, volatility: 15.5, description: "Entire US stock market" },
  VOO: { name: "Vanguard S&P 500", category: "US Large Cap", expenseRatio: 0.03, avgReturn: 10.2, volatility: 14.8, description: "500 largest US companies" },
  VUG: { name: "Vanguard Growth", category: "US Growth", expenseRatio: 0.04, avgReturn: 12.1, volatility: 18.2, description: "US growth stocks" },
  VTV: { name: "Vanguard Value", category: "US Value", expenseRatio: 0.04, avgReturn: 9.5, volatility: 14.0, description: "US value stocks" },
  
  // Tech/Growth
  QQQ: { name: "Invesco Nasdaq 100", category: "Tech", expenseRatio: 0.20, avgReturn: 14.5, volatility: 20.5, description: "Top 100 Nasdaq stocks" },
  
  // International
  VXUS: { name: "Vanguard Total International", category: "International", expenseRatio: 0.07, avgReturn: 6.8, volatility: 16.0, description: "Non-US stocks" },
  VEA: { name: "Vanguard Developed Markets", category: "Developed Intl", expenseRatio: 0.05, avgReturn: 7.2, volatility: 15.5, description: "Developed markets ex-US" },
  VWO: { name: "Vanguard Emerging Markets", category: "Emerging", expenseRatio: 0.08, avgReturn: 8.5, volatility: 22.0, description: "Emerging market stocks" },
  
  // Bonds
  BND: { name: "Vanguard Total Bond", category: "US Bonds", expenseRatio: 0.03, avgReturn: 4.5, volatility: 5.5, description: "US investment-grade bonds" },
  AGG: { name: "iShares Core US Aggregate Bond", category: "US Bonds", expenseRatio: 0.03, avgReturn: 4.3, volatility: 5.2, description: "Broad US bond market" },
  TLT: { name: "iShares 20+ Year Treasury", category: "Long-Term Bonds", expenseRatio: 0.15, avgReturn: 5.2, volatility: 14.0, description: "Long-term US treasuries" },
  TIP: { name: "iShares TIPS Bond", category: "Inflation Protected", expenseRatio: 0.19, avgReturn: 4.0, volatility: 6.5, description: "Inflation-protected bonds" },
  
  // Thematic
  ICLN: { name: "iShares Global Clean Energy", category: "Clean Energy", expenseRatio: 0.40, avgReturn: 11.0, volatility: 28.0, description: "Global clean energy stocks" },
  VHT: { name: "Vanguard Health Care", category: "Healthcare", expenseRatio: 0.10, avgReturn: 11.5, volatility: 14.5, description: "Healthcare sector" },
  VNQ: { name: "Vanguard Real Estate", category: "Real Estate", expenseRatio: 0.12, avgReturn: 9.0, volatility: 18.0, description: "REITs and real estate" },
  SCHD: { name: "Schwab US Dividend Equity", category: "Dividends", expenseRatio: 0.06, avgReturn: 10.8, volatility: 13.5, description: "High dividend stocks" },
  
  // More Thematic ETFs
  BOTZ: { name: "Global X Robotics & AI", category: "AI & Robotics", expenseRatio: 0.68, avgReturn: 13.5, volatility: 25.0, description: "AI and robotics companies" },
  ARKK: { name: "ARK Innovation", category: "Disruptive Innovation", expenseRatio: 0.75, avgReturn: 15.0, volatility: 35.0, description: "Disruptive innovation stocks" },
  XLF: { name: "Financial Select Sector SPDR", category: "Financials", expenseRatio: 0.09, avgReturn: 9.8, volatility: 18.5, description: "US financial sector" },
  XLE: { name: "Energy Select Sector SPDR", category: "Energy", expenseRatio: 0.09, avgReturn: 8.5, volatility: 25.0, description: "US energy sector" },
  SOXX: { name: "iShares Semiconductor", category: "Semiconductors", expenseRatio: 0.35, avgReturn: 18.0, volatility: 28.0, description: "Semiconductor companies" },
  IBB: { name: "iShares Biotechnology", category: "Biotech", expenseRatio: 0.44, avgReturn: 10.5, volatility: 22.0, description: "Biotechnology companies" },
  TAN: { name: "Invesco Solar", category: "Solar Energy", expenseRatio: 0.67, avgReturn: 12.0, volatility: 32.0, description: "Solar energy companies" },
  LIT: { name: "Global X Lithium & Battery", category: "EV & Batteries", expenseRatio: 0.75, avgReturn: 14.0, volatility: 30.0, description: "Lithium and battery tech" },
  HACK: { name: "ETFMG Prime Cyber Security", category: "Cybersecurity", expenseRatio: 0.60, avgReturn: 12.5, volatility: 20.0, description: "Cybersecurity companies" },
  BLOK: { name: "Amplify Blockchain", category: "Blockchain", expenseRatio: 0.71, avgReturn: 11.0, volatility: 35.0, description: "Blockchain technology" },
};

// Risk profile allocations
const RISK_PROFILES: Record<string, {
  name: string;
  description: string;
  color: string;
  allocation: Record<string, number>;
}> = {
  conservative: {
    name: "Conservative",
    description: "Focus on capital preservation with steady income",
    color: "text-blue-500",
    allocation: { VTI: 20, VXUS: 10, BND: 50, TIP: 20 },
  },
  moderate: {
    name: "Moderate",
    description: "Balanced growth with moderate risk",
    color: "text-green-500",
    allocation: { VTI: 35, VXUS: 15, BND: 35, TIP: 15 },
  },
  balanced: {
    name: "Balanced",
    description: "Equal focus on growth and stability",
    color: "text-yellow-500",
    allocation: { VTI: 45, VXUS: 20, BND: 25, TIP: 10 },
  },
  growth: {
    name: "Growth",
    description: "Higher growth potential with more volatility",
    color: "text-orange-500",
    allocation: { VTI: 50, QQQ: 15, VXUS: 20, BND: 15 },
  },
  aggressive: {
    name: "Aggressive",
    description: "Maximum growth, higher risk tolerance",
    color: "text-red-500",
    allocation: { VTI: 45, QQQ: 25, VXUS: 20, VUG: 5, BND: 5 },
  },
};

// Quiz questions
const QUIZ_QUESTIONS = [
  {
    id: 1,
    question: "What is your investment time horizon?",
    options: [
      { text: "Less than 3 years", score: 1 },
      { text: "3-5 years", score: 2 },
      { text: "5-10 years", score: 3 },
      { text: "10-20 years", score: 4 },
      { text: "20+ years", score: 5 },
    ],
  },
  {
    id: 2,
    question: "If your portfolio dropped 20% in a month, you would:",
    options: [
      { text: "Sell everything immediately", score: 1 },
      { text: "Sell some to reduce risk", score: 2 },
      { text: "Hold and wait it out", score: 3 },
      { text: "Buy a little more", score: 4 },
      { text: "Buy aggressively - great opportunity!", score: 5 },
    ],
  },
  {
    id: 3,
    question: "What is your primary investment goal?",
    options: [
      { text: "Preserve my capital at all costs", score: 1 },
      { text: "Generate steady income", score: 2 },
      { text: "Balanced growth and income", score: 3 },
      { text: "Long-term growth", score: 4 },
      { text: "Maximum growth, I can handle volatility", score: 5 },
    ],
  },
  {
    id: 4,
    question: "How would you describe your investment knowledge?",
    options: [
      { text: "Beginner - just starting out", score: 2 },
      { text: "Basic - understand stocks and bonds", score: 3 },
      { text: "Intermediate - familiar with ETFs and diversification", score: 4 },
      { text: "Advanced - understand market cycles and risk", score: 5 },
    ],
  },
  {
    id: 5,
    question: "What percentage of your savings is this investment?",
    options: [
      { text: "Over 75% - most of my savings", score: 1 },
      { text: "50-75%", score: 2 },
      { text: "25-50%", score: 3 },
      { text: "10-25%", score: 4 },
      { text: "Less than 10% - I have other investments", score: 5 },
    ],
  },
];

interface SimulationResult {
  years: number[];
  median: number[];
  percentile10: number[];
  percentile25: number[];
  percentile75: number[];
  percentile90: number[];
  finalMedian: number;
  final10: number;
  final90: number;
  probabilityOf500k: number;
  probabilityOf1m: number;
}

export default function ETFRecommender() {
  const navigate = useNavigate();
  const [step, setStep] = useState<"quiz" | "results" | "simulate">("quiz");
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<number[]>([]);
  const [riskProfile, setRiskProfile] = useState<string | null>(null);
  const [customAllocation, setCustomAllocation] = useState<Record<string, number>>({});
  const [thematicAddons, setThematicAddons] = useState<string[]>([]);
  
  // Simulation params
  const [initialInvestment, setInitialInvestment] = useState(10000);
  const [monthlyContribution, setMonthlyContribution] = useState(500);
  const [years, setYears] = useState(20);
  const [simResult, setSimResult] = useState<SimulationResult | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) navigate("/");
    });
  }, [navigate]);

  const handleAnswer = (score: number) => {
    const newAnswers = [...answers, score];
    setAnswers(newAnswers);
    
    if (currentQuestion < QUIZ_QUESTIONS.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      // Calculate risk profile
      const totalScore = newAnswers.reduce((a, b) => a + b, 0);
      const maxScore = QUIZ_QUESTIONS.length * 5;
      const percentage = totalScore / maxScore;
      
      let profile: string;
      if (percentage < 0.3) profile = "conservative";
      else if (percentage < 0.45) profile = "moderate";
      else if (percentage < 0.6) profile = "balanced";
      else if (percentage < 0.75) profile = "growth";
      else profile = "aggressive";
      
      setRiskProfile(profile);
      setCustomAllocation({ ...RISK_PROFILES[profile].allocation });
      setStep("results");
    }
  };

  const toggleThematic = (etf: string) => {
    if (thematicAddons.includes(etf)) {
      setThematicAddons(thematicAddons.filter(e => e !== etf));
    } else {
      setThematicAddons([...thematicAddons, etf]);
    }
  };

  const getFinalAllocation = (): Record<string, number> => {
    const base = { ...customAllocation };
    const thematicWeight = 5; // Each thematic adds 5%
    
    if (thematicAddons.length > 0) {
      // Reduce base allocations proportionally
      const totalReduction = thematicAddons.length * thematicWeight;
      const scaleFactor = (100 - totalReduction) / 100;
      
      Object.keys(base).forEach(key => {
        base[key] = Math.round(base[key] * scaleFactor);
      });
      
      // Add thematic ETFs
      thematicAddons.forEach(etf => {
        base[etf] = thematicWeight;
      });
    }
    
    return base;
  };

  const runMonteCarloSimulation = () => {
    setIsSimulating(true);
    
    const allocation = getFinalAllocation();
    const numSimulations = 1000;
    const allResults: number[][] = [];
    
    // Calculate portfolio expected return and volatility
    let portfolioReturn = 0;
    let portfolioVolatility = 0;
    
    Object.entries(allocation).forEach(([etf, weight]) => {
      const data = ETF_DATA[etf];
      if (data) {
        portfolioReturn += (data.avgReturn / 100) * (weight / 100);
        // Simplified: assume correlations of 0.5 between assets
        portfolioVolatility += Math.pow((data.volatility / 100) * (weight / 100), 2);
      }
    });
    portfolioVolatility = Math.sqrt(portfolioVolatility) * 1.3; // Adjust for correlations
    
    // Run simulations
    for (let sim = 0; sim < numSimulations; sim++) {
      const yearlyValues: number[] = [initialInvestment];
      let currentValue = initialInvestment;
      
      for (let year = 1; year <= years; year++) {
        // Generate random return using normal distribution (Box-Muller)
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        const yearReturn = portfolioReturn + z * portfolioVolatility;
        
        // Apply return and add contributions
        currentValue = currentValue * (1 + yearReturn);
        currentValue += monthlyContribution * 12;
        
        yearlyValues.push(Math.max(0, currentValue));
      }
      
      allResults.push(yearlyValues);
    }
    
    // Calculate percentiles for each year
    const yearsArray = Array.from({ length: years + 1 }, (_, i) => i);
    const median: number[] = [];
    const percentile10: number[] = [];
    const percentile25: number[] = [];
    const percentile75: number[] = [];
    const percentile90: number[] = [];
    
    yearsArray.forEach((_, yearIndex) => {
      const valuesAtYear = allResults.map(sim => sim[yearIndex]).sort((a, b) => a - b);
      
      percentile10.push(valuesAtYear[Math.floor(numSimulations * 0.1)]);
      percentile25.push(valuesAtYear[Math.floor(numSimulations * 0.25)]);
      median.push(valuesAtYear[Math.floor(numSimulations * 0.5)]);
      percentile75.push(valuesAtYear[Math.floor(numSimulations * 0.75)]);
      percentile90.push(valuesAtYear[Math.floor(numSimulations * 0.9)]);
    });
    
    // Calculate probabilities
    const finalValues = allResults.map(sim => sim[sim.length - 1]);
    const prob500k = finalValues.filter(v => v >= 500000).length / numSimulations * 100;
    const prob1m = finalValues.filter(v => v >= 1000000).length / numSimulations * 100;
    
    setTimeout(() => {
      setSimResult({
        years: yearsArray,
        median,
        percentile10,
        percentile25,
        percentile75,
        percentile90,
        finalMedian: median[median.length - 1],
        final10: percentile10[percentile10.length - 1],
        final90: percentile90[percentile90.length - 1],
        probabilityOf500k: Math.round(prob500k),
        probabilityOf1m: Math.round(prob1m),
      });
      setIsSimulating(false);
      setStep("simulate");
    }, 1000);
  };

  const formatCurrency = (value: number) => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(1)}M`;
    }
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(value);
  };

  const restartQuiz = () => {
    setStep("quiz");
    setCurrentQuestion(0);
    setAnswers([]);
    setRiskProfile(null);
    setCustomAllocation({});
    setThematicAddons([]);
    setSimResult(null);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-md">
        <div className="max-w-4xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate("/dashboard")}
              className="p-2 rounded-lg hover:bg-muted transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-primary/10">
                <PieChart className="w-4 h-4 text-primary" />
              </div>
              <span className="font-display text-sm font-semibold">ETF Recommender</span>
            </div>
          </div>
          {step !== "quiz" && (
            <button
              onClick={restartQuiz}
              className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
            >
              <RefreshCw className="w-3 h-3" />
              Retake Quiz
            </button>
          )}
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-6 py-8">
        {/* Quiz Step */}
        {step === "quiz" && (
          <div className="max-w-xl mx-auto">
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
                <Target className="w-8 h-8 text-primary" />
              </div>
              <h1 className="font-display text-3xl font-bold mb-2">Find Your Risk Profile</h1>
              <p className="text-muted-foreground">
                Answer {QUIZ_QUESTIONS.length} quick questions to get personalized ETF recommendations
              </p>
            </div>

            {/* Progress */}
            <div className="mb-8">
              <div className="flex justify-between text-sm text-muted-foreground mb-2">
                <span>Question {currentQuestion + 1} of {QUIZ_QUESTIONS.length}</span>
                <span>{Math.round(((currentQuestion) / QUIZ_QUESTIONS.length) * 100)}% complete</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${((currentQuestion) / QUIZ_QUESTIONS.length) * 100}%` }}
                />
              </div>
            </div>

            {/* Question */}
            <div className="glass-card rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-6">
                {QUIZ_QUESTIONS[currentQuestion].question}
              </h2>
              <div className="space-y-3">
                {QUIZ_QUESTIONS[currentQuestion].options.map((option, index) => (
                  <button
                    key={index}
                    onClick={() => handleAnswer(option.score)}
                    className="w-full text-left p-4 rounded-lg border border-border hover:border-primary hover:bg-primary/5 transition-colors flex items-center justify-between group"
                  >
                    <span>{option.text}</span>
                    <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Results Step */}
        {step === "results" && riskProfile && (
          <div>
            <div className="text-center mb-8">
              <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4`}>
                {riskProfile === "conservative" && <Shield className="w-8 h-8 text-blue-500" />}
                {riskProfile === "moderate" && <Shield className="w-8 h-8 text-green-500" />}
                {riskProfile === "balanced" && <Target className="w-8 h-8 text-yellow-500" />}
                {riskProfile === "growth" && <TrendingUp className="w-8 h-8 text-orange-500" />}
                {riskProfile === "aggressive" && <Zap className="w-8 h-8 text-red-500" />}
              </div>
              <h1 className="font-display text-3xl font-bold mb-2">
                Your Profile: <span className={RISK_PROFILES[riskProfile].color}>{RISK_PROFILES[riskProfile].name}</span>
              </h1>
              <p className="text-muted-foreground">
                {RISK_PROFILES[riskProfile].description}
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Recommended Allocation */}
              <div className="glass-card rounded-xl p-6">
                <h2 className="font-display text-lg font-semibold mb-4 flex items-center gap-2">
                  <PieChart className="w-5 h-5 text-primary" />
                  Recommended Portfolio
                </h2>
                
                {/* Pie Chart Visualization */}
                <div className="relative w-48 h-48 mx-auto mb-6">
                  <svg viewBox="0 0 100 100" className="transform -rotate-90">
                    {(() => {
                      const allocation = getFinalAllocation();
                      let currentAngle = 0;
                      const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];
                      
                      return Object.entries(allocation).map(([etf, weight], index) => {
                        const angle = (weight / 100) * 360;
                        const startAngle = currentAngle;
                        currentAngle += angle;
                        
                        const x1 = 50 + 40 * Math.cos((startAngle * Math.PI) / 180);
                        const y1 = 50 + 40 * Math.sin((startAngle * Math.PI) / 180);
                        const x2 = 50 + 40 * Math.cos(((startAngle + angle) * Math.PI) / 180);
                        const y2 = 50 + 40 * Math.sin(((startAngle + angle) * Math.PI) / 180);
                        const largeArc = angle > 180 ? 1 : 0;
                        
                        return (
                          <path
                            key={etf}
                            d={`M 50 50 L ${x1} ${y1} A 40 40 0 ${largeArc} 1 ${x2} ${y2} Z`}
                            fill={colors[index % colors.length]}
                            className="hover:opacity-80 transition-opacity"
                          />
                        );
                      });
                    })()}
                  </svg>
                </div>

                {/* ETF List */}
                <div className="space-y-2">
                  {Object.entries(getFinalAllocation()).map(([etf, weight], index) => {
                    const data = ETF_DATA[etf];
                    const colors = ['bg-blue-500', 'bg-green-500', 'bg-yellow-500', 'bg-red-500', 'bg-purple-500', 'bg-pink-500', 'bg-cyan-500', 'bg-lime-500'];
                    
                    return (
                      <div key={etf} className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50">
                        <div className={`w-3 h-3 rounded ${colors[index % colors.length]}`} />
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-semibold">{etf}</span>
                            <span className="text-xs text-muted-foreground">{data?.name}</span>
                          </div>
                        </div>
                        <span className="font-semibold">{weight}%</span>
                      </div>
                    );
                  })}
                </div>

                {/* Portfolio Stats - key forces re-render when themes change */}
                <div key={`stats-${thematicAddons.join('-')}`} className="mt-4 pt-4 border-t border-border grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Expected Return</div>
                    <div className="font-semibold text-success text-lg">
                      {(() => {
                        const allocation = getFinalAllocation();
                        let ret = 0;
                        Object.entries(allocation).forEach(([etf, weight]) => {
                          const data = ETF_DATA[etf];
                          if (data) ret += data.avgReturn * (weight / 100);
                        });
                        return `${ret.toFixed(1)}%`;
                      })()}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Expense Ratio</div>
                    <div className="font-semibold text-lg">
                      {(() => {
                        const allocation = getFinalAllocation();
                        let exp = 0;
                        Object.entries(allocation).forEach(([etf, weight]) => {
                          const data = ETF_DATA[etf];
                          if (data) exp += data.expenseRatio * (weight / 100);
                        });
                        return `${exp.toFixed(2)}%`;
                      })()}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Volatility</div>
                    <div className="font-semibold text-lg">
                      {(() => {
                        const allocation = getFinalAllocation();
                        let vol = 0;
                        Object.entries(allocation).forEach(([etf, weight]) => {
                          const data = ETF_DATA[etf];
                          if (data) vol += Math.pow(data.volatility * (weight / 100), 2);
                        });
                        return `${Math.sqrt(vol).toFixed(1)}%`;
                      })()}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Holdings</div>
                    <div className="font-semibold text-lg">
                      {Object.keys(getFinalAllocation()).length} ETFs
                    </div>
                  </div>
                </div>
              </div>

              {/* Thematic Add-ons */}
              <div className="space-y-6">
                <div className="glass-card rounded-xl p-6 max-h-[400px] overflow-y-auto">
                  <h2 className="font-display text-lg font-semibold mb-4 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-primary" />
                    Add Thematic Exposure (Optional)
                  </h2>
                  <p className="text-sm text-muted-foreground mb-4">
                    Each adds 5% allocation. Select up to 4 themes.
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { etf: "BOTZ", label: "AI & Robotics", icon: "🤖", ret: 13.5 },
                      { etf: "SOXX", label: "Semiconductors", icon: "💻", ret: 18.0 },
                      { etf: "ARKK", label: "Innovation", icon: "🚀", ret: 15.0 },
                      { etf: "ICLN", label: "Clean Energy", icon: "🌱", ret: 11.0 },
                      { etf: "TAN", label: "Solar", icon: "☀️", ret: 12.0 },
                      { etf: "LIT", label: "EV & Batteries", icon: "🔋", ret: 14.0 },
                      { etf: "VHT", label: "Healthcare", icon: "🏥", ret: 11.5 },
                      { etf: "IBB", label: "Biotech", icon: "🧬", ret: 10.5 },
                      { etf: "XLF", label: "Financials", icon: "🏦", ret: 9.8 },
                      { etf: "XLE", label: "Energy", icon: "⛽", ret: 8.5 },
                      { etf: "VNQ", label: "Real Estate", icon: "🏠", ret: 9.0 },
                      { etf: "SCHD", label: "Dividends", icon: "💰", ret: 10.8 },
                      { etf: "HACK", label: "Cybersecurity", icon: "🔐", ret: 12.5 },
                      { etf: "BLOK", label: "Blockchain", icon: "⛓️", ret: 11.0 },
                    ].map(({ etf, label, icon, ret }) => (
                      <button
                        key={etf}
                        onClick={() => toggleThematic(etf)}
                        disabled={thematicAddons.length >= 4 && !thematicAddons.includes(etf)}
                        className={`p-2.5 rounded-lg border text-left transition-colors ${
                          thematicAddons.includes(etf)
                            ? "border-primary bg-primary/10"
                            : thematicAddons.length >= 4
                            ? "border-border opacity-50 cursor-not-allowed"
                            : "border-border hover:border-primary/50"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-lg">{icon}</span>
                          <div className="flex-1 min-w-0">
                            <div className="font-semibold text-xs truncate">{label}</div>
                            <div className="text-xs text-muted-foreground">{etf} • {ret}%</div>
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                  {thematicAddons.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-border">
                      <div className="text-xs text-muted-foreground">
                        Selected: {thematicAddons.join(", ")} ({thematicAddons.length * 5}% total)
                      </div>
                    </div>
                  )}
                </div>

                {/* Simulation Params */}
                <div className="glass-card rounded-xl p-6">
                  <h2 className="font-display text-lg font-semibold mb-4 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-primary" />
                    Test This Portfolio
                  </h2>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                        <DollarSign className="w-4 h-4" />
                        Initial Investment
                      </label>
                      <input
                        type="number"
                        value={initialInvestment}
                        onChange={(e) => setInitialInvestment(Number(e.target.value))}
                        className="w-full px-4 py-2 bg-muted rounded-lg focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                    </div>
                    
                    <div>
                      <label className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                        <DollarSign className="w-4 h-4" />
                        Monthly Contribution
                      </label>
                      <input
                        type="number"
                        value={monthlyContribution}
                        onChange={(e) => setMonthlyContribution(Number(e.target.value))}
                        className="w-full px-4 py-2 bg-muted rounded-lg focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                    </div>
                    
                    <div>
                      <label className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                        <Calendar className="w-4 h-4" />
                        Time Horizon: {years} years
                      </label>
                      <input
                        type="range"
                        min="5"
                        max="40"
                        value={years}
                        onChange={(e) => setYears(Number(e.target.value))}
                        className="w-full"
                      />
                    </div>
                    
                    <button
                      onClick={runMonteCarloSimulation}
                      disabled={isSimulating}
                      className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-primary-foreground rounded-lg hover:opacity-90 disabled:opacity-50"
                    >
                      {isSimulating ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Running 1,000 Simulations...
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" />
                          Run Monte Carlo Simulation
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Simulation Results */}
        {step === "simulate" && simResult && (
          <div>
            <div className="text-center mb-8">
              <h1 className="font-display text-3xl font-bold mb-2">Monte Carlo Results</h1>
              <p className="text-muted-foreground">
                Based on 1,000 simulated scenarios using historical returns
              </p>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-3 gap-4 mb-8">
              <div className="glass-card rounded-xl p-4 text-center border-l-4 border-red-500">
                <div className="text-sm text-muted-foreground mb-1">Worst Case (10th %ile)</div>
                <div className="text-2xl font-bold text-red-500">
                  {formatCurrency(simResult.final10)}
                </div>
              </div>
              <div className="glass-card rounded-xl p-4 text-center border-l-4 border-primary">
                <div className="text-sm text-muted-foreground mb-1">Expected (Median)</div>
                <div className="text-2xl font-bold text-primary">
                  {formatCurrency(simResult.finalMedian)}
                </div>
              </div>
              <div className="glass-card rounded-xl p-4 text-center border-l-4 border-green-500">
                <div className="text-sm text-muted-foreground mb-1">Best Case (90th %ile)</div>
                <div className="text-2xl font-bold text-green-500">
                  {formatCurrency(simResult.final90)}
                </div>
              </div>
            </div>

            {/* Fan Chart */}
            <div className="glass-card rounded-xl p-6 mb-8">
              <h2 className="font-display text-lg font-semibold mb-4">Projected Growth Range</h2>
              <div className="relative h-64">
                {/* Y-axis labels */}
                <div className="absolute left-0 top-0 bottom-8 w-16 flex flex-col justify-between text-xs text-muted-foreground">
                  <span>{formatCurrency(simResult.percentile90[simResult.percentile90.length - 1])}</span>
                  <span>{formatCurrency(simResult.median[Math.floor(simResult.median.length / 2)])}</span>
                  <span>{formatCurrency(initialInvestment)}</span>
                </div>
                
                {/* Chart Area */}
                <div className="ml-16 h-full relative">
                  <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full">
                    {/* 10-90 percentile band */}
                    <polygon
                      points={simResult.years.map((_, i) => {
                        const x = (i / (simResult.years.length - 1)) * 100;
                        const maxVal = simResult.percentile90[simResult.percentile90.length - 1];
                        const y90 = 100 - (simResult.percentile90[i] / maxVal) * 90;
                        return `${x},${y90}`;
                      }).join(' ') + ' ' + [...simResult.years].reverse().map((_, i) => {
                        const idx = simResult.years.length - 1 - i;
                        const x = (idx / (simResult.years.length - 1)) * 100;
                        const maxVal = simResult.percentile90[simResult.percentile90.length - 1];
                        const y10 = 100 - (simResult.percentile10[idx] / maxVal) * 90;
                        return `${x},${y10}`;
                      }).join(' ')}
                      fill="rgba(59, 130, 246, 0.1)"
                    />
                    
                    {/* 25-75 percentile band */}
                    <polygon
                      points={simResult.years.map((_, i) => {
                        const x = (i / (simResult.years.length - 1)) * 100;
                        const maxVal = simResult.percentile90[simResult.percentile90.length - 1];
                        const y75 = 100 - (simResult.percentile75[i] / maxVal) * 90;
                        return `${x},${y75}`;
                      }).join(' ') + ' ' + [...simResult.years].reverse().map((_, i) => {
                        const idx = simResult.years.length - 1 - i;
                        const x = (idx / (simResult.years.length - 1)) * 100;
                        const maxVal = simResult.percentile90[simResult.percentile90.length - 1];
                        const y25 = 100 - (simResult.percentile25[idx] / maxVal) * 90;
                        return `${x},${y25}`;
                      }).join(' ')}
                      fill="rgba(59, 130, 246, 0.2)"
                    />
                    
                    {/* Median line */}
                    <polyline
                      points={simResult.years.map((_, i) => {
                        const x = (i / (simResult.years.length - 1)) * 100;
                        const maxVal = simResult.percentile90[simResult.percentile90.length - 1];
                        const y = 100 - (simResult.median[i] / maxVal) * 90;
                        return `${x},${y}`;
                      }).join(' ')}
                      fill="none"
                      stroke="#3b82f6"
                      strokeWidth="2"
                    />
                  </svg>
                  
                  {/* X-axis labels */}
                  <div className="flex justify-between mt-2 text-xs text-muted-foreground">
                    <span>Year 0</span>
                    <span>Year {Math.floor(years / 2)}</span>
                    <span>Year {years}</span>
                  </div>
                </div>
              </div>
              
              {/* Legend */}
              <div className="flex gap-6 justify-center mt-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-blue-500/10 border border-blue-500/30" />
                  <span className="text-muted-foreground">10th-90th percentile</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-blue-500/20 border border-blue-500/50" />
                  <span className="text-muted-foreground">25th-75th percentile</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-1 bg-blue-500 rounded" />
                  <span className="text-muted-foreground">Median</span>
                </div>
              </div>
            </div>

            {/* Probability Cards */}
            <div className="grid grid-cols-2 gap-4 mb-8">
              <div className="glass-card rounded-xl p-6 text-center">
                <div className="text-4xl font-bold text-primary mb-2">{simResult.probabilityOf500k}%</div>
                <div className="text-muted-foreground">Probability of reaching $500,000</div>
                <div className="mt-3 h-2 bg-muted rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-primary transition-all"
                    style={{ width: `${simResult.probabilityOf500k}%` }}
                  />
                </div>
              </div>
              <div className="glass-card rounded-xl p-6 text-center">
                <div className="text-4xl font-bold text-success mb-2">{simResult.probabilityOf1m}%</div>
                <div className="text-muted-foreground">Probability of reaching $1,000,000</div>
                <div className="mt-3 h-2 bg-muted rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-success transition-all"
                    style={{ width: `${simResult.probabilityOf1m}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-4 justify-center">
              <button
                onClick={() => setStep("results")}
                className="px-6 py-3 bg-muted text-foreground rounded-lg hover:bg-muted/80 flex items-center gap-2"
              >
                <ArrowLeft className="w-4 h-4" />
                Adjust Portfolio
              </button>
              <button
                onClick={runMonteCarloSimulation}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:opacity-90 flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                Run Again
              </button>
            </div>

            {/* Disclaimer */}
            <div className="mt-8 p-4 rounded-lg bg-muted/50 text-center">
              <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                <Info className="w-4 h-4" />
                <span>
                  Simulations use historical average returns and volatility. Past performance does not guarantee future results.
                  This is for educational purposes only and not financial advice.
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
