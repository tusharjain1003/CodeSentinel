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
  comments?: unknown[];
  created_at?: string;
};

type ReviewStore = {
  reviews: Review[];
  evalMetrics: Record<string, Metrics>;
  fetchReviews: () => Promise<void>;
  fetchEvalMetrics: () => Promise<void>;
};

const emptyMetrics = { precision: 0, recall: 0, f1: 0, quality: 0 };

export const useReviewStore = create<ReviewStore>((set) => ({
  reviews: [],
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
  fetchEvalMetrics: async () => {
    const response = await fetch("/api/eval/metrics");
    const data = await response.json();
    set({ evalMetrics: data.models ?? data });
  }
}));
