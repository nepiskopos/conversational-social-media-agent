-- Org memory, accounts and the tool-call trace, with row-level security per org.
-- Applied by agent/db.migrate() (scripts/migrate.py) as the owner role, after LangGraph's
-- checkpoint tables.
-- Every org table is visible only to a transaction whose app.org_id setting names that
-- org (agent/db.py sets it), so a query that forgets its WHERE clause still can't
-- reach another org. FORCE applies the policies to the tables' owner too.

CREATE TABLE orgs (
    org_id text PRIMARY KEY CHECK (org_id ~ '^[a-z0-9]+(-[a-z0-9]+)*$'),
    name text NOT NULL DEFAULT '',
    website text NOT NULL DEFAULT '',
    research jsonb,
    org_brief text NOT NULL DEFAULT '',
    research_updated_at text NOT NULL DEFAULT '',
    content_pillars jsonb,
    -- Highest post number handed out, so a deleted post's id is never reused.
    last_post_number integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE posts (
    org_id text NOT NULL REFERENCES orgs ON DELETE CASCADE,
    post_id text NOT NULL CHECK (post_id ~ '^post_[0-9]+$'),
    -- Ledger order: the order posts were saved in.
    seq bigint GENERATED ALWAYS AS IDENTITY,
    status text NOT NULL
        CHECK (status IN ('suggested', 'draft', 'approved', 'posted', 'planned', 'rejected')),
    platform text NOT NULL DEFAULT '',
    content text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    -- Optional fields that vary by post (title, pillar, scheduled_for, parent_id, visual, ...).
    details jsonb NOT NULL DEFAULT '{}',
    performance jsonb,
    PRIMARY KEY (org_id, post_id)
);
CREATE INDEX posts_org_seq ON posts (org_id, seq);

CREATE TABLE preferences (
    org_id text NOT NULL REFERENCES orgs ON DELETE CASCADE,
    key text NOT NULL,
    seq bigint GENERATED ALWAYS AS IDENTITY,
    value jsonb NOT NULL,
    quote text,
    learned_at timestamptz,
    PRIMARY KEY (org_id, key)
);

CREATE TABLE voice_examples (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id text NOT NULL REFERENCES orgs ON DELETE CASCADE,
    content text NOT NULL,
    source text NOT NULL DEFAULT '',
    added_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX voice_examples_org ON voice_examples (org_id, id);

CREATE TABLE tool_calls (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    org_id text NOT NULL REFERENCES orgs ON DELETE CASCADE,
    called_at timestamptz NOT NULL DEFAULT now(),
    tool text NOT NULL,
    input jsonb NOT NULL,
    result_preview text NOT NULL,
    latency_ms integer NOT NULL
);
CREATE INDEX tool_calls_org ON tool_calls (org_id, id);

-- Accounts are not per org: logging in happens before an org is chosen.
CREATE TABLE users (
    user_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username text NOT NULL UNIQUE CHECK (username ~ '^[a-z0-9._@+-]{1,64}$'),
    password_hash text NOT NULL,
    failed_logins integer NOT NULL DEFAULT 0,
    locked_until timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE memberships (
    user_id bigint NOT NULL REFERENCES users ON DELETE CASCADE,
    org_id text NOT NULL REFERENCES orgs ON DELETE CASCADE,
    PRIMARY KEY (user_id, org_id)
);

DO $$
DECLARE
    org_table text;
BEGIN
    FOREACH org_table IN ARRAY ARRAY['orgs', 'posts', 'preferences', 'voice_examples', 'tool_calls']
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', org_table);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', org_table);
        EXECUTE format(
            'CREATE POLICY org_scope ON %I USING (org_id = current_setting(''app.org_id'', true)) '
            'WITH CHECK (org_id = current_setting(''app.org_id'', true))',
            org_table
        );
    END LOOP;
    -- Conversation threads are "{org_id}" or "{org_id}:{rest}", e.g. "{org_id}:web-user-7"
    -- (agent/assistant.py), so the part before the first ":" is the org.
    FOREACH org_table IN ARRAY ARRAY['checkpoints', 'checkpoint_blobs', 'checkpoint_writes']
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', org_table);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', org_table);
        EXECUTE format(
            'CREATE POLICY org_scope ON %I '
            'USING (split_part(thread_id, '':'', 1) = current_setting(''app.org_id'', true)) '
            'WITH CHECK (split_part(thread_id, '':'', 1) = current_setting(''app.org_id'', true))',
            org_table
        );
    END LOOP;
END
$$;
