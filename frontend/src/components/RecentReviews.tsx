import { ExternalLink, PanelRightOpen } from "lucide-react";
import type { Review } from "../state/reviewStore";

type Props = {
  reviews: Review[];
  onSelect: (review: Review) => void;
};

export function RecentReviews({ reviews, onSelect }: Props) {
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
              <div className="reviewActions">
                {review.id ? (
                  <button
                    className="iconButton"
                    type="button"
                    onClick={() => onSelect(review)}
                    aria-label="Open review details"
                    title="Open review details"
                  >
                    <PanelRightOpen size={18} />
                  </button>
                ) : null}
                <a href={review.pr_url} target="_blank" rel="noreferrer" aria-label="Open PR">
                  <ExternalLink size={18} />
                </a>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
