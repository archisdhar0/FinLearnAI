import { Sparkles, TrendingUp } from "lucide-react";

export function HeroSection() {
  return (
    <section className="py-16 animate-fade-up">
      <div className="flex items-center gap-2 mb-6">
        <div className="p-2 rounded-xl bg-primary/10 glow-gold">
          <TrendingUp className="w-6 h-6 text-primary" />
        </div>
        <span className="text-sm font-medium text-primary tracking-wide uppercase">FinLearn AI</span>
      </div>

      <h1 className="font-display text-5xl lg:text-6xl font-bold mb-6 leading-tight">
        Learn to invest
        <br />
        <span className="text-gradient-gold">with confidence</span>
      </h1>

      <p className="text-lg text-muted-foreground max-w-xl mb-8 leading-relaxed">
        Master the fundamentals of investing through structured lessons and an
        AI tutor that answers your questions in real time.
      </p>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-sm text-muted-foreground glass-card rounded-full px-4 py-2">
          <Sparkles className="w-4 h-4 text-primary" />
          AI-powered learning assistant
        </div>
      </div>
    </section>
  );
}
