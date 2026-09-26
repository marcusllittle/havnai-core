# Clerk lifecycle and account access

`POST /v2/auth/clerk/webhook` accepts Clerk events with a verified Svix signature
over the original request bytes. Configure a dedicated endpoint for each Clerk
instance and coordinator database. Its private service environment needs:

```dotenv
HAVNAI_CLERK_ISSUER=https://<the-same-issuer-used-for-account-auth>
HAVNAI_CLERK_INSTANCE_ID=ins_<expected-instance>
CLERK_WEBHOOK_SIGNING_SECRET=<this-endpoint's-signing-secret>
```

Subscribe to `user.created`, `user.updated`, `user.deleted`, `session.ended`,
`session.removed`, and `session.revoked`. Install `server/requirements.txt` before
starting the service. Svix 2.x verifies signatures without parsing JSON; this
handler parses the body only after verification. Reject missing/invalid signatures,
expired delivery signatures, oversized payloads, and other Clerk instances.

## Access and retention

- Login still provisions accounts synchronously from a verified issuer/subject.
  It does not wait for a creation webhook or match users by email.
- Ban state follows the signed User object's `updated_at`. Older events cannot
  override newer state; ties favor a ban. A later unban restores provider access,
  but cannot clear an independent local account suspension.
- Deletion leaves a permanent identity tombstone, including when received before
  the first login. Creation/update replays cannot recreate that identity.
- Session termination records are scoped to issuer and session ID. They block
  that session while other valid sessions remain usable. User deletion/ban blocks
  all sessions for that identity.
- Account provisioning, wallet proofs, and authenticated studio/account routes
  check provider state. Receipts, credits, jobs, and wallet links remain owned by
  the same account. Provider lifecycle events never transfer or erase them.
- State changes and the event acknowledgement commit atomically. Duplicate event
  IDs with matching payload hashes are harmless; conflicting reuse is rejected.
  The audit stores no email address, profile payload, or bearer token.

## Delivery and acceptance

Webhooks are asynchronous. Revocation takes effect locally after verified delivery;
it does not retroactively cancel a request already authorized or a running job.
Without delivery, existing bearer tokens can remain usable until their short
expiry (the API permits at most 120 seconds; Clerk development currently issues
60-second tokens). Configure delivery failure alerts and replay failed messages
from Clerk. Before enabling production accounts, demonstrate deletion, banning,
session revocation, and replay against the deployed endpoint and confirm access
denial on account and studio routes.

`tests/test_account_lifecycle.py` verifies real Svix signatures and real SQLite
transactions, including duplicate delivery, ordering, rollback, instance isolation,
and HTTP denial for otherwise valid signed bearer tokens. Studio tests verify the
same revocation check gates job submission/recovery. These are not evidence of a
live Clerk endpoint delivery. The isolated preview disables webhook credentials
by default so it cannot inherit another environment's signing secret.

References: [Clerk webhook payloads and delivery](https://clerk.com/docs/guides/development/webhooks/overview),
[Svix signature verification](https://docs.svix.com/receiving/verifying-payloads/how).
