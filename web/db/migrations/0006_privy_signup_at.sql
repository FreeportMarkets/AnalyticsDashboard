-- Keep the Privy signup date alongside each mirrored wallet identity so
-- trade rows can identify recent signups without a live Privy API request.
ALTER TABLE privy_identities ADD COLUMN IF NOT EXISTS created_at timestamptz;
