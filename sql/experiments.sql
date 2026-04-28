-- experiment bookkeeping so sweeps do not turn into mystery folders

CREATE TABLE IF NOT EXISTS experiment_runs (
  id            BIGSERIAL PRIMARY KEY,
  name          TEXT NOT NULL,
  git_sha       TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  notes         TEXT
);

CREATE INDEX IF NOT EXISTS idx_exp_runs_time ON experiment_runs (created_at DESC);

CREATE TABLE IF NOT EXISTS hparams (
  run_id     BIGINT NOT NULL REFERENCES experiment_runs (id) ON DELETE CASCADE,
  key        TEXT NOT NULL,
  value_json JSONB NOT NULL,
  PRIMARY KEY (run_id, key)
);

CREATE TABLE IF NOT EXISTS run_metrics (
  run_id     BIGINT NOT NULL REFERENCES experiment_runs (id) ON DELETE CASCADE,
  step       INTEGER NOT NULL,
  name       TEXT NOT NULL,
  value      DOUBLE PRECISION NOT NULL,
  logged_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, step, name)
);

CREATE INDEX IF NOT EXISTS idx_run_metrics_run ON run_metrics (run_id, logged_at);

CREATE TABLE IF NOT EXISTS checkpoints (
  id          BIGSERIAL PRIMARY KEY,
  run_id      BIGINT NOT NULL REFERENCES experiment_runs (id) ON DELETE CASCADE,
  path        TEXT NOT NULL,
  step        INTEGER NOT NULL,
  is_promoted BOOLEAN NOT NULL DEFAULT false,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ckpt_run ON checkpoints (run_id, step);

-- partial index: promoted ckpts are rare; lookups should stay instant
CREATE INDEX IF NOT EXISTS idx_ckpt_promoted
  ON checkpoints (run_id)
  WHERE is_promoted;

-- tags so you can filter ui without parsing free text notes
CREATE TABLE IF NOT EXISTS run_tags (
  run_id BIGINT NOT NULL REFERENCES experiment_runs (id) ON DELETE CASCADE,
  tag    TEXT NOT NULL,
  PRIMARY KEY (run_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_run_tags_tag ON run_tags (tag);
