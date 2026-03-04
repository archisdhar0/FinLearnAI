import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { apiCall } from "@/lib/api";
import {
  ArrowLeft,
  Target,
  PieChart,
  Zap,
  Shield,
  TrendingUp,
  DollarSign,
  Calendar,
  Info,
} from "lucide-react";

const ETF_META: Record<
  string,
  { name: string; category: string; description: string }
> = {
  VTI: {
    name: "Vanguard Total Stock Market",
    category: "US Stocks",
    description: "Entire US stock market",
  },
  VOO: {
    name: "Vanguard S&P 500",
    category: "US Large Cap",
    description: "500 largest US companies",
  },
  VXUS: {
    name: "Vanguard Total International",
    category: "International",
    description: "Non-US stocks",
  },
  QQQ: {
    name: "Invesco Nasdaq 100",
    category: "Tech / Growth",
    description: "Top 100 Nasdaq stocks",
  },
  BND: {
    name: "Vanguard Total Bond",
    category: "US Bonds",
    description: "US investment-grade bonds",
  },
  AGG: {
    name: "iShares Core US Aggregate Bond",
    category: "US Bonds",
    description: "Broad US bond market",
  },
  TIP: {
    name: "iShares TIPS Bond",
    category: "Inflation Protected",
    description: "Inflation-protected US bonds",
  },
  VNQ: {
    name: "Vanguard Real Estate",
    category: "Real Estate",
    description: "US REITs and real estate",
  },
};

type PrimaryGoal =
  | "capital_preservation"
  | "income"
  | "balanced"
  | "growth"
  | "max_growth";

interface GAAllocationSummary {
  risk_score: number;
  inferred_profile: string;
  expected_return_annual_pct: number;
  expected_volatility_annual_pct: number;
  expense_ratio_pct: number;
  equity_weight_pct: number;
}

interface GAAllocationResponse {
  allocations: Record<string, number>;
  summary: GAAllocationSummary;
}

type Step = "quiz" | "results";

interface QuizQuestion {
  id: number;
  question: string;
  helper?: string;
  type: "scale" | "goal" | "horizon";
  options: { text: string; value: number | PrimaryGoal }[];
}

const QUIZ_QUESTIONS: QuizQuestion[] = [
  {
    id: 1,
    type: "horizon",
    question: "What is your investment time horizon for this money?",
    helper: "When will you likely need to spend most of this portfolio?",
    options: [
      { text: "Less than 3 years", value: 1 },
      { text: "3–5 years", value: 2 },
      { text: "5–10 years", value: 3 },
      { text: "10–20 years", value: 4 },
      { text: "20+ years", value: 5 },
    ],
  },
  {
    id: 2,
    type: "scale",
    question: "If your portfolio dropped 20% in a month, you would:",
    options: [
      { text: "Sell everything immediately", value: 1 },
      { text: "Sell some to reduce risk", value: 2 },
      { text: "Hold and wait it out", value: 3 },
      { text: "Buy a little more", value: 4 },
      { text: "Buy aggressively – great opportunity!", value: 5 },
    ],
  },
  {
    id: 3,
    type: "goal",
    question: "What is your primary investment goal for this portfolio?",
    options: [
      { text: "Preserve capital above all else", value: "capital_preservation" },
      { text: "Generate steady income", value: "income" },
      { text: "Balanced growth and income", value: "balanced" },
      { text: "Long-term growth", value: "growth" },
      {
        text: "Maximum growth – I can handle big swings",
        value: "max_growth",
      },
    ],
  },
  {
    id: 4,
    type: "scale",
    question: "How would you describe your investing knowledge?",
    options: [
      { text: "Beginner – just getting started", value: 1 },
      { text: "Basic – understand stocks and bonds", value: 2 },
      { text: "Intermediate – familiar with ETFs & diversification", value: 3 },
      { text: "Advanced – understand risk, cycles, and asset classes", value: 4 },
      { text: "Expert – very comfortable with markets", value: 5 },
    ],
  },
  {
    id: 5,
    type: "scale",
    question: "How stable is your income and job situation?",
    options: [
      { text: "Very unstable – at risk or irregular", value: 1 },
      { text: "Somewhat unstable", value: 2 },
      { text: "Moderately stable", value: 3 },
      { text: "Stable", value: 4 },
      { text: "Very stable – highly predictable", value: 5 },
    ],
  },
  {
    id: 6,
    type: "scale",
    question: "How much investment risk are you willing to take for higher returns?",
    options: [
      { text: "Very low – I lose sleep over losses", value: 1 },
      { text: "Low", value: 2 },
      { text: "Moderate", value: 3 },
      { text: "High", value: 4 },
      { text: "Very high – big swings are fine", value: 5 },
    ],
  },
];

export default function GAAllocation() {
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>("quiz");
  const [currentQuestion, setCurrentQuestion] = useState(0);

  const [timeHorizonScore, setTimeHorizonScore] = useState<number | null>(null);
  const [drawdownTolerance, setDrawdownTolerance] = useState<number | null>(null);
  const [primaryGoal, setPrimaryGoal] = useState<PrimaryGoal | null>(null);
  const [knowledge, setKnowledge] = useState<number | null>(null);
  const [incomeStability, setIncomeStability] = useState<number | null>(null);
  const [riskTolerance, setRiskTolerance] = useState<number | null>(null);

  const [allocations, setAllocations] = useState<Record<string, number> | null>(
    null,
  );
  const [summary, setSummary] = useState<GAAllocationSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) navigate("/");
    });
  }, [navigate]);

  const handleAnswer = async (value: number | PrimaryGoal) => {
    const question = QUIZ_QUESTIONS[currentQuestion];

    if (question.type === "horizon" && typeof value === "number") {
      setTimeHorizonScore(value);
    } else if (question.type === "goal" && typeof value === "string") {
      setPrimaryGoal(value as PrimaryGoal);
    } else if (question.type === "scale" && typeof value === "number") {
      if (question.id === 2) setDrawdownTolerance(value);
      else if (question.id === 4) setKnowledge(value);
      else if (question.id === 5) setIncomeStability(value);
      else if (question.id === 6) setRiskTolerance(value);
    }

    if (currentQuestion < QUIZ_QUESTIONS.length - 1) {
      setCurrentQuestion((q) => q + 1);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const horizonScore = timeHorizonScore ?? (question.type === "horizon" && typeof value === "number" ? value : 3);
      const drawdown =
        drawdownTolerance ??
        (question.id === 2 && typeof value === "number" ? value : 3);
      const goal =
        primaryGoal ??
        ((question.type === "goal" && typeof value === "string"
          ? value
          : "balanced") as PrimaryGoal);
      const knowledgeScore =
        knowledge ??
        (question.id === 4 && typeof value === "number" ? value : 3);
      const incomeScore =
        incomeStability ??
        (question.id === 5 && typeof value === "number" ? value : 3);
      const riskScore =
        riskTolerance ??
        (question.id === 6 && typeof value === "number" ? value : 3);

      const horizonMap = [2, 4, 7, 15, 25];
      const time_horizon_years =
        horizonMap[horizonScore - 1] ?? 10;

      const body = {
        time_horizon_years,
        risk_tolerance: riskScore,
        drawdown_tolerance: drawdown,
        investment_knowledge: knowledgeScore,
        income_stability: incomeScore,
        primary_goal: goal,
      };

      const response = await apiCall<GAAllocationResponse>(
        "/api/ai/asset-allocation/etf-ga",
        {
          method: "POST",
          body: JSON.stringify(body),
        },
      );

      setAllocations(response.allocations || {});
      setSummary(response.summary);
      setStep("results");
    } catch (err) {
      console.error(err);
      setError(
        "We couldn't compute the GA allocation. Please try again in a moment.",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const restart = () => {
    setStep("quiz");
    setCurrentQuestion(0);
    setTimeHorizonScore(null);
    setDrawdownTolerance(null);
    setPrimaryGoal(null);
    setKnowledge(null);
    setIncomeStability(null);
    setRiskTolerance(null);
    setAllocations(null);
    setSummary(null);
    setError(null);
    setIsLoading(false);
  };

  const current = QUIZ_QUESTIONS[currentQuestion];
  const progress = Math.round(
    ((currentQuestion) / QUIZ_QUESTIONS.length) * 100,
  );

  return (
    <div className="min-h-screen bg-background">
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
              <span className="font-display text-sm font-semibold">
                GA ETF Allocation
              </span>
            </div>
          </div>
          {step === "results" && (
            <button
              onClick={restart}
              className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
            >
              <Zap className="w-3 h-3" />
              New Allocation
            </button>
          )}
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-6 py-8">
        {step === "quiz" && (
          <div className="max-w-xl mx-auto">
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
                <Target className="w-8 h-8 text-primary" />
              </div>
              <h1 className="font-display text-3xl font-bold mb-2">
                Build Your ETF Mix
              </h1>
              <p className="text-muted-foreground">
                Answer {QUIZ_QUESTIONS.length} questions and we&apos;ll design an ETF
                allocation using a genetic algorithm.
              </p>
            </div>

            <div className="mb-8">
              <div className="flex justify-between text-sm text-muted-foreground mb-2">
                <span>
                  Question {currentQuestion + 1} of {QUIZ_QUESTIONS.length}
                </span>
                <span>{progress}% complete</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            <div className="glass-card rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-2">{current.question}</h2>
              {current.helper && (
                <p className="text-sm text-muted-foreground mb-4">
                  {current.helper}
                </p>
              )}

              <div className="space-y-3">
                {current.options.map((opt, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleAnswer(opt.value)}
                    className="w-full text-left p-4 rounded-lg border border-border hover:border-primary hover:bg-primary/5 transition-colors flex items-center justify-between group"
                  >
                    <span>{opt.text}</span>
                    <Shield className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                  </button>
                ))}
              </div>

              {error && (
                <p className="mt-4 text-sm text-destructive">{error}</p>
              )}
            </div>
          </div>
        )}

        {step === "results" && allocations && summary && (
          <div className="space-y-8">
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4">
                <PieChart className="w-8 h-8 text-primary" />
              </div>
              <h1 className="font-display text-3xl font-bold mb-2">
                Your GA-Based ETF Allocation
              </h1>
              <p className="text-muted-foreground">
                Inferred profile:{" "}
                <span className="font-semibold">
                  {summary.inferred_profile}
                </span>{" "}
                • Expected return{" "}
                <span className="font-semibold">
                  {summary.expected_return_annual_pct.toFixed(1)}%/yr
                </span>{" "}
                • Volatility{" "}
                <span className="font-semibold">
                  {summary.expected_volatility_annual_pct.toFixed(1)}%/yr
                </span>
              </p>
            </div>

            <div className="glass-card rounded-xl p-6">
              <h2 className="font-display text-lg font-semibold mb-4 flex items-center gap-2">
                <PieChart className="w-5 h-5 text-primary" />
                ETF Allocation
              </h2>

              <div className="space-y-2">
                {Object.entries(allocations)
                  .sort(([, a], [, b]) => (b ?? 0) - (a ?? 0))
                  .map(([ticker, weight]) => {
                    const meta = ETF_META[ticker];
                    return (
                      <div
                        key={ticker}
                        className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50"
                      >
                        <div className="w-10 font-mono font-semibold">
                          {ticker}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between gap-2">
                            <div className="min-w-0">
                              <div className="text-sm font-medium truncate">
                                {meta?.name ?? "Unknown ETF"}
                              </div>
                              <div className="text-xs text-muted-foreground truncate">
                                {meta?.category ?? "ETF"}{" "}
                                {meta?.description ? `• ${meta.description}` : ""}
                              </div>
                            </div>
                            <div className="flex items-center gap-2 w-40">
                              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-primary"
                                  style={{ width: `${weight}%` }}
                                />
                              </div>
                              <span className="w-12 text-right text-sm font-semibold">
                                {weight.toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="glass-card rounded-xl p-4 flex items-center gap-3">
                <TrendingUp className="w-6 h-6 text-primary" />
                <div>
                  <div className="text-xs text-muted-foreground">
                    Expected Return (annual)
                  </div>
                  <div className="text-lg font-semibold">
                    {summary.expected_return_annual_pct.toFixed(1)}%
                  </div>
                </div>
              </div>
              <div className="glass-card rounded-xl p-4 flex items-center gap-3">
                <Shield className="w-6 h-6 text-yellow-500" />
                <div>
                  <div className="text-xs text-muted-foreground">
                    Volatility (annual)
                  </div>
                  <div className="text-lg font-semibold">
                    {summary.expected_volatility_annual_pct.toFixed(1)}%
                  </div>
                </div>
              </div>
              <div className="glass-card rounded-xl p-4 flex items-center gap-3">
                <DollarSign className="w-6 h-6 text-emerald-500" />
                <div>
                  <div className="text-xs text-muted-foreground">
                    Fee Drag (expense ratio)
                  </div>
                  <div className="text-lg font-semibold">
                    {summary.expense_ratio_pct.toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4 p-4 rounded-lg bg-muted/60 text-sm flex items-start gap-2">
              <Info className="w-4 h-4 mt-0.5 text-muted-foreground" />
              <div>
                <p className="text-muted-foreground">
                  This allocation is generated by a genetic algorithm that
                  searches over ETF weight combinations to balance expected
                  return, risk, and fees while matching your risk profile.
                  Results are for education only and are not financial advice.
                </p>
              </div>
            </div>
          </div>
        )}

        {isLoading && step === "quiz" && (
          <div className="mt-6 text-center text-sm text-muted-foreground flex items-center justify-center gap-2">
            <Calendar className="w-4 h-4 animate-spin" />
            Computing GA allocation...
          </div>
        )}
      </div>
    </div>
  );
}

