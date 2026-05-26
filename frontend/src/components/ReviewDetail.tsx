import { ThumbsDown, ThumbsUp } from "lucide-react";
import type { Review } from "../state/reviewStore";

type Props = {
  review: Review | null;
  onFeedback: (commentIdx: number, rating: -1 | 0 | 1) => Promise<void>;
};

const severityLabels: Record<string, string> = {
  critical: "Critical",
  major: "Major",
  minor: "Minor",
  nit: "Nit"
};

export function ReviewDetail({ review, onFeedback }: Props) {
  if (!review) {
    return null;
  }

  const comments = review.comments ?? [];

  return (
    <section className="panel">
      <div className="panelHeader detailHeader">
        <div>
          <h2>Review Detail</h2>
          <p>
            {review.repo} PR #{review.pr_number}
          </p>
        </div>
        <span>{review.model_used ?? "unknown model"}</span>
      </div>

      {Object.keys(review.timing_ms ?? {}).length ? (
        <div className="timingGrid">
          {Object.entries(review.timing_ms ?? {}).map(([agent, timing]) => (
            <div key={agent}>
              <span>{agent}</span>
              <strong>{timing} ms</strong>
            </div>
          ))}
        </div>
      ) : null}

      {comments.length === 0 ? (
        <div className="empty">No comments recorded for this review.</div>
      ) : (
        <div className="commentList">
          {comments.map((comment, index) => (
            <article className="commentItem" key={`${comment.file_path}-${index}`}>
              <div className="commentMeta">
                <strong>{severityLabels[comment.severity] ?? comment.severity}</strong>
                <span>{comment.category}</span>
                <span>
                  {comment.file_path}:{comment.line_start}
                </span>
                <span>{Math.round(comment.confidence * 100)}%</span>
              </div>
              <p>{comment.message}</p>
              {comment.suggestion ? <blockquote>{comment.suggestion}</blockquote> : null}
              {review.id ? (
                <div className="feedbackActions">
                  <button
                    className="iconButton"
                    type="button"
                    onClick={() => onFeedback(index, 1)}
                    aria-label="Mark helpful"
                    title="Mark helpful"
                  >
                    <ThumbsUp size={16} />
                  </button>
                  <button
                    className="iconButton"
                    type="button"
                    onClick={() => onFeedback(index, -1)}
                    aria-label="Mark unhelpful"
                    title="Mark unhelpful"
                  >
                    <ThumbsDown size={16} />
                  </button>
                </div>
              ) : null}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
