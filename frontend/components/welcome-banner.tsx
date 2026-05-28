"use client";

type WelcomeBannerProps = {
  onNewChat?: () => void;
};

export function WelcomeBanner({ onNewChat }: WelcomeBannerProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center">
      <div className="max-w-3xl">
        <div className="space-y-3">
          <h1 className="text-2xl md:text-3xl lg:text-4xl font-bold text-chart-3">
            🤖 TechnoBot | An AI Chatbot Mentor for Technical Analysis!
          </h1>
        </div>

        <div className="space-y-4 mt-6">
          <p className="text-base leading-relaxed text-chart-2 italic">
            Analyze market structure, momentum, and volatility with data-driven
            insights. Ask guided questions to deepen your understanding of price
            action or upload your own CSV data for custom analysis.
          </p>

          <p className="text-sm text-muted-foreground italic">
            Ask me about any instrument or upload historical data to get
            started!
          </p>
        </div>
      </div>
    </div>
  );
}
