#!/bin/sh
# Runs once, when the postgres container starts with an empty data volume.
# Creates Mackie's two roles and its database. mackie_owner owns the schema and runs
# migrations (scripts/migrate.py); mackie_app is what Mackie connects as: it can read
# and write rows, nothing else, and row-level security applies to it. Neither is a
# superuser. The superuser (POSTGRES_USER) is used only here.
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
  -v owner_password="$MACKIE_OWNER_PASSWORD" -v app_password="$MACKIE_APP_PASSWORD" <<'SQL'
CREATE ROLE mackie_owner LOGIN PASSWORD :'owner_password';
CREATE ROLE mackie_app LOGIN PASSWORD :'app_password';
CREATE DATABASE mackie OWNER mackie_owner;
REVOKE ALL ON DATABASE mackie FROM PUBLIC;
GRANT CONNECT ON DATABASE mackie TO mackie_app;
SQL
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname mackie <<'SQL'
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
ALTER SCHEMA public OWNER TO mackie_owner;
SQL
