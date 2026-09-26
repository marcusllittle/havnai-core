# Adult content moderation and public isolation policy

Work item: HAVN-41, under HAVN-14 and HAVN-11.

This policy defines how HavnAI treats adult or NSFW-capable generation before
commercial launch. It is an operating and product policy, not proof that the
server-side enforcement layer is complete. HAVN-39 owns implementation, HAVN-40
owns automated public-surface tests, and HAVN-42 owns live production
verification.

## Launch boundary

Adult content may exist only as private, explicitly requested account content
when it is otherwise legal and allowed by this policy. Adult content must never
appear on public HavnAI surfaces during the HAVN-11 commercial launch.

The current launch does not include a public adult marketplace, adult Discover
feed, adult creator profile, adult playlist surface, or adult-branded product.
Any future public adult product requires separate architecture, separate
branding/domain review, age-access controls, payment/provider review, public
policy pages, support workflow, legal review, and an explicit launch decision.

Content involving minors, suggested minors, child sexual exploitation, or
ambiguous youth in a sexual context is always blocked. This is not private adult
content and is not eligible for publication, appeal as adult content, or
operator bypass.

## Classification

Treat an item as adult-restricted when any of these signals apply:

- The prompt, negative prompt, upload metadata, model, workflow, or operator
  review indicates adult, explicit, erotic, nude, pornographic, hentai, or NSFW
  content.
- The selected model or workflow is tagged adult, NSFW, hentai, or equivalent in
  the model registry or operator catalog.
- A generated artifact, thumbnail, cover, preview, title, description, lyrics,
  playlist entry, listing, profile module, or social preview is reported or
  reviewed as adult.

When signals conflict, use the most restrictive public-surface decision until a
moderator resolves the item. The system must prefer a safe private restriction
over accidental public exposure.

## Public surfaces

Adult-restricted content must be excluded or rejected server-side from every
public or anonymous surface, including:

- Discover, search, recommendations, homepage modules, creator profiles, public
  libraries, public playlists, and playlist embeds.
- Marketplace listings, listing previews, sale pages, and public gallery APIs.
- Public audio, image, video, thumbnail, cover, preview, waveform, transcript,
  metadata, and direct media routes.
- Sitemap, Open Graph, social cards, cache keys, analytics previews, share URLs,
  public webhooks, and user-visible logs or error details.

UI hiding is not sufficient. Public route handlers and query builders must
consult the same adult-restriction policy and return safe 403 or 404 responses
without exposing prompts, filenames, storage paths, model names, classifier
details, private account identifiers, or internal moderation notes.

## Owner access

The owning account may access private adult-restricted content only through
authenticated account routes while the account and artifact remain active and no
hold, takedown, or legal restriction applies. Cross-account access is denied.

Deleting, restoring, purging, refunding, disputing, or auditing adult-restricted
content follows the normal account lifecycle contracts, with the adult flag and
moderation events retained in the audit trail. Receipts, credit ledger rows,
settlement records, and anchored records remain immutable where required, but
customer-visible copies must not disclose restricted prompt text or private
media.

## Publication and listing

Adult-restricted artifacts cannot be published, listed for sale, added to public
playlists, embedded in public profiles, or attached to public templates during
this launch. The server should reject the operation with product-safe copy such
as:

> This item can stay private in your account, but it cannot be published on
> HavnAI public surfaces.

Do not expose the exact classifier rule, banned term, storage path, model file,
or operator note in the user-facing response. Log the internal reason with a
correlation ID for support and audit review.

## Reporting

Every public content surface should provide a report path or support link that
captures:

- Public URL or item ID.
- Reporter account ID when signed in, or an anonymous report marker.
- Report reason, free-text context, timestamp, user agent, and correlation ID
  when available.
- Snapshot of public metadata needed for review, excluding secrets and private
  media access tokens.

Reports create an audit event and place the target item into a moderation review
queue. For suspected child sexual exploitation or imminent harm, operators must
escalate through the legal and safety process immediately and preserve relevant
records according to law and company policy.

## Takedown and review

Operators may restrict, unpublish, delist, hide from public surfaces, suspend
sharing, or disable media access while a report is reviewed. Takedown actions
must be idempotent and recorded with actor, reason category, target IDs,
timestamp, previous state, new state, and correlation ID.

When a takedown affects a marketplace listing, playlist, profile, or publication,
the public reference must disappear immediately. Ownership, receipts, ledgers,
credit records, and audit history remain intact. Refund, dispute, or customer
credit decisions are handled through the account payment and support process,
not by silently editing ledger history.

Appeals may restore private owner access or remove an adult-restricted flag only
after review. Appeals must not republish the item automatically. Any return to a
public surface requires a fresh publication/listing request and a current policy
check.

## Audit and retention

Moderation events are append-only. Records must include enough information to
explain who acted, what changed, why the action occurred, and which customer or
public surfaces were affected. Store private prompt text, media paths, and
operator notes only in access-controlled systems. Do not copy them into Jira,
public logs, support emails, browser-visible errors, analytics labels, or
customer-facing receipts.

Adult restriction state must travel with the artifact through deletion,
restoration, publication attempts, playlist edits, marketplace operations,
legacy import review, backup restore, and account migration. Backup and restore
drills must verify that restriction flags, moderation events, and public
exclusions survive recovery.

## Launch checklist

HAVN-14 can cite this policy only after the remaining implementation and evidence
lanes are complete:

- HAVN-39: durable adult metadata and server-side public isolation policy.
- HAVN-40: route and public-surface tests for Discover, search, profiles,
  playlists, media, thumbnails, marketplace, and public APIs.
- HAVN-42: live production verification with exact URLs, status codes,
  screenshots or logs, and redacted account details.

Until those lanes are complete or explicitly waived as launch risk, HAVN-14 and
HAVN-11 remain open.
