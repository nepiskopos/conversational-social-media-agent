-- Publishing approved posts to the org's social media accounts (agent/publishing.py).

-- The account of each platform an org publishes to, and the permissions (OAuth scopes) it
-- was granted. No access token is stored here: a live publisher reads the token for an
-- account from a secret store, so a database dump never holds the keys to an org's pages.
CREATE TABLE platform_accounts (
    org_id text NOT NULL REFERENCES orgs ON DELETE CASCADE,
    platform text NOT NULL CHECK (platform ~ '^[a-z0-9_]+$'),
    -- The platform's id of the page or profile posted to, e.g. a LinkedIn organization URN.
    account_id text NOT NULL CHECK (account_id ~ '^\S{1,200}$'),
    account_name text NOT NULL DEFAULT '',
    scopes text[] NOT NULL DEFAULT '{}',
    connected_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (org_id, platform)
);

-- One row per post and platform, claimed ('publishing') before the platform is called, so
-- two sessions can never publish a post to a platform twice. Not tied to posts: an import
-- that replaces the ledger keeps the record of what went out (post ids are never reused).
CREATE TABLE publications (
    org_id text NOT NULL REFERENCES orgs ON DELETE CASCADE,
    post_id text NOT NULL CHECK (post_id ~ '^post_[0-9]+$'),
    platform text NOT NULL,
    state text NOT NULL CHECK (state IN ('publishing', 'published', 'failed')),
    account_id text NOT NULL,
    external_id text NOT NULL DEFAULT '',
    url text NOT NULL DEFAULT '',
    error text NOT NULL DEFAULT '',
    simulated boolean NOT NULL DEFAULT false,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (org_id, post_id, platform)
);

DO $$
DECLARE
    org_table text;
BEGIN
    FOREACH org_table IN ARRAY ARRAY['platform_accounts', 'publications']
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', org_table);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', org_table);
        EXECUTE format(
            'CREATE POLICY org_scope ON %I USING (org_id = current_setting(''app.org_id'', true)) '
            'WITH CHECK (org_id = current_setting(''app.org_id'', true))',
            org_table
        );
    END LOOP;
END
$$;
