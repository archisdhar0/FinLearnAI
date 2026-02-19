import { useNavigate } from "react-router-dom";
import { BookOpen, Search, Settings } from "lucide-react";

const topics = [
  {
    id: "foundations",
    icon: BookOpen,
    title: "The Foundation",
    description: "Core concepts every first-time investor should know. What is investing, how markets work, risk basics, and getting started.",
    level: "Beginner",
    lessons: 7,
  },
  {
    id: "investor-insight",
    icon: Search,
    title: "Investor Insight",
    description: "Deeper thinking about markets and behavior. Market psychology, economic indicators, and understanding what moves prices.",
    level: "Intermediate",
    lessons: 5,
  },
  {
    id: "applied-investing",
    icon: Settings,
    title: "Applied Investing",
    description: "Practical steps to build and manage your portfolio. Asset allocation, rebalancing, tax strategies, and long-term planning.",
    level: "Advanced",
    lessons: 6,
  },
];

const levelColors: Record<string, string> = {
  Beginner: "bg-success/20 text-success",
  Intermediate: "bg-info/20 text-info",
  Advanced: "bg-primary/20 text-primary",
};

export function LessonGrid() {
  const navigate = useNavigate();

  const handleTopicClick = (topicId: string) => {
    navigate(`/learn/${topicId}`);
  };

  return (
    <section className="py-12">
      <h2 className="font-display text-3xl font-bold mb-2">Learning Paths</h2>
      <p className="text-muted-foreground mb-8">Master investing step by step</p>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
        {topics.map((topic, i) => (
          <div
            key={topic.title}
            onClick={() => handleTopicClick(topic.id)}
            className={`glass-card rounded-xl p-6 hover:border-primary/40 transition-all duration-300 cursor-pointer group animate-fade-up animate-delay-${(i % 4) * 100}`}
          >
            <div className="flex items-start justify-between mb-4">
              <div className="p-2.5 rounded-lg bg-primary/10 text-primary group-hover:glow-gold transition-all">
                <topic.icon className="w-5 h-5" />
              </div>
              <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${levelColors[topic.level]}`}>
                {topic.level}
              </span>
            </div>
            <h3 className="font-display text-lg font-semibold mb-2 group-hover:text-primary transition-colors">
              {topic.title}
            </h3>
            <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
              {topic.description}
            </p>
            <div className="text-xs text-muted-foreground">
              {topic.lessons} lessons
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
