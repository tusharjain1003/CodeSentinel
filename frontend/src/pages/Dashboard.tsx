import { ShieldCheck } from "lucide-react";
import { useEffect } from "react";
import { ManualReviewForm } from "../components/ManualReviewForm";
import { MetricTable } from "../components/MetricTable";
import { RecentReviews } from "../components/RecentReviews";
import { useReviewStore } from "../state/reviewStore";

export function App() {
  const { reviews, evalMetrics, fetchReviews, fetchEvalMetrics } = useReviewStore();

  useEffect(() => {
    fetchReviews();
    fetchEvalMetrics();
  }, [fetchReviews, fetchEvalMetrics]);

  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <ShieldCheck size={28} />
          <div>
            <h1>CodeSentinel</h1>
            <p>AI PR reviews with specialist agents and structured outputs</p>
          </div>
        </div>
      </header>

      <section className="layout">
        <div className="mainColumn">
          <MetricTable metrics={evalMetrics} />
          <RecentReviews reviews={reviews} />
        </div>
        <aside>
          <ManualReviewForm />
        </aside>
      </section>
    </main>
  );
}
