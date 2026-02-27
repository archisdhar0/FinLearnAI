import { Sparkles, TrendingUp, Clock } from "lucide-react";

export function HeroSection() {
  return (
    <section className="py-10 animate-fade-up">
      <div className="flex items-center gap-2 mb-6">
        <div className="p-2 rounded-xl bg-primary/10 glow-gold">
          <TrendingUp className="w-6 h-6 text-primary" />
        </div>
        <span className="text-sm font-medium text-primary tracking-wide uppercase">FinLearn AI</span>
      </div>

      <h1 className="font-display text-4xl lg:text-5xl font-bold mb-4 leading-tight">
        Stop waiting.
        <br />
        <span className="text-gradient-gold">Start investing today.</span>
      </h1>

      <p className="text-lg text-muted-foreground max-w-2xl mb-6 leading-relaxed">
        You don't need thousands of dollars or a finance degree. You just need to start.
        Learn the fundamentals in <span className="text-foreground font-medium">3 structured modules</span> with 
        an AI tutor that explains everything in plain English.
      </p>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2 text-sm text-muted-foreground glass-card rounded-full px-4 py-2">
          <Sparkles className="w-4 h-4 text-primary" />
          AI-powered learning
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground glass-card rounded-full px-4 py-2">
          <Clock className="w-4 h-4 text-info" />
          ~2 hours total
        </div>
        <div className="flex items-center gap-2 text-sm text-success glass-card rounded-full px-4 py-2 border-success/30">
          100% free
        </div>
      </div>
    </section>
  );
}
