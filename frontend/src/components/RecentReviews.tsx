import { ExternalLink } from "lucide-react";
import type { Review } from "../state/reviewStore";

type Props = {
  reviews: Review[];
};

export function RecentReviews({ reviews }: Props) {
  return (
    <section className="panel">
      <div className="panelHeader">
        <h2>Recent Reviews</h2>
      </div>
      {reviews.length === 0 ? (
        <div className="empty">No reviews yet.</div>
      ) : (
        <div className="reviewList">
          {reviews.map((review) => (
            <article className="reviewItem" key={review.id ?? review.pr_url}>
              <div>
                <strong>{review.repo}</strong>
                <span>PR #{review.pr_number}</span>
              </div>
              <a href={review.pr_url} target="_blank" rel="noreferrer" aria-label="Open PR">
                <ExternalLink size={18} />
              </a>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
