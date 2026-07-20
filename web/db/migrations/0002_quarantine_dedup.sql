-- Dedup quarantine rows so a permanently-wedged window doesn't re-quarantine
-- the same poisoned row every sync tick forever. Re-running a sync over any
-- range must not change the resulting data (idempotency requirement).

ALTER TABLE quarantine ADD COLUMN IF NOT EXISTS item_hash text;

CREATE UNIQUE INDEX IF NOT EXISTS quarantine_source_item_hash_idx ON quarantine (source, item_hash);
