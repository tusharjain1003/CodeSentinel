import { create } from "zustand";

export type Metrics = {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  quality: number | null;
  status?: string;
};

export type Review = {
  id?: string;
  pr_url: string;
  repo: string;
  pr_number: number;
  model_used?: string;
  comments?: ReviewComment[];
  timing_ms?: Record<string, number>;
  token_cost?: Record<string, unknown>;
  created_at?: string;
};

export type ReviewComment = {
  category: string;
  severity: string;
  file_path: string;
  line_start: number;
  line_end: number;
  message: string;
  suggestion?: string | null;
  confidence: number;
};

type ReviewStore = {
  reviews: Review[];
  selectedReview: Review | null;
  evalMetrics: Record<string, Metrics>;
  fetchReviews: () => Promise<void>;
  fetchReview: (reviewId: string) => Promise<void>;
  fetchEvalMetrics: () => Promise<void>;
  submitFeedback: (
    reviewId: string,
    commentIdx: number,
    rating: -1 | 0 | 1,
    correction?: string
  ) => Promise<void>;
};

const emptyMetrics = { precision: 0, recall: 0, f1: 0, quality: 0 };

export const useReviewStore = create<ReviewStore>((set) => ({
  reviews: [],
  selectedReview: null,
  evalMetrics: {
    finetuned: emptyMetrics,
    base: emptyMetrics,
    gpt4o: emptyMetrics,
    groq: emptyMetrics,
    heuristic: emptyMetrics
  },
  fetchReviews: async () => {
    const response = await fetch("/api/reviews");
    const data = await response.json();
    set({ reviews: data.reviews ?? [] });
  },
  fetchReview: async (reviewId: string) => {
    const response = await fetch(`/api/reviews/${reviewId}`);
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    set({ selectedReview: data });
  },
  fetchEvalMetrics: async () => {
    const response = await fetch("/api/eval/metrics");
    const data = await response.json();
    set({ evalMetrics: data.models ?? data });
  },
  submitFeedback: async (reviewId, commentIdx, rating, correction) => {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review_id: reviewId, comment_idx: commentIdx, rating, correction })
    });
  }
}));
