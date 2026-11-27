-- Read-only role for the agent. Run with an admin user: `make db-role`.
-- IMPORTANT: re-run after every `make sql` in Hotel-Booking-Demand, because that
-- script drops and recreates the `bookings` materialized view (and its grants).
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_ro') THEN
        CREATE ROLE agent_ro LOGIN PASSWORD 'agent_ro';   -- change for anything non-local
    END IF;
END
$$;

GRANT CONNECT ON DATABASE hotel TO agent_ro;
GRANT USAGE ON SCHEMA public TO agent_ro;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM agent_ro;
GRANT SELECT ON bookings TO agent_ro;              -- the ONLY object it can read

ALTER ROLE agent_ro SET statement_timeout = '5s';
ALTER ROLE agent_ro SET default_transaction_read_only = on;
