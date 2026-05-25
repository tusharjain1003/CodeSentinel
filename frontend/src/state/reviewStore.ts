import { create } from "zustand";

export type Metrics = {
  precision: number;
  recall: number;
  f1: number;
  quality: number;
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
    gpt4o: emptyMetrics
  },
  fetchReviews: async () => {
    const response = await fetch("/api/reviews");
    const data = await response.json();
    set({ reviews: data.reviews ?? [] });
  },
  fetchEvalMetrics: async () => {
    const response = await fetch("/api/eval/metrics");
    const data = await response.json();
    set({ evalMetrics: data });
  }
}));
