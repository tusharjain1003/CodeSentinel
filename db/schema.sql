CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS reviews (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pr_url      TEXT NOT NULL,
    repo        TEXT NOT NULL,
    pr_number   INTEGER NOT NULL,
    diff_hash   TEXT NOT NULL,
    model_used  TEXT NOT NULL,
    comments    JSONB NOT NULL,
    timing_ms   JSONB,
    token_cost  JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS review_feedback (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    review_id   UUID REFERENCES reviews(id),
    comment_idx INTEGER NOT NULL,
    rating      SMALLINT CHECK (rating IN (-1, 0, 1)),
    correction  TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reviews_diff_hash ON reviews(diff_hash);
CREATE INDEX IF NOT EXISTS idx_reviews_repo_pr ON reviews(repo, pr_number);
