# HAVN-47 mixed-model worker drill

`scripts/mixed_model_worker_drill.py` submits private account jobs through
`/v2/jobs` and polls them to a terminal state. The default, and `--dry-run`,
print the plan and do not contact the coordinator. `--execute` is the live run.
It refuses to submit when the quoted balance cannot cover the whole plan.

`/v2` accepts only a Clerk session JWT. `server/account_auth.py` rejects a token
whose `exp - iat` is greater than 120 seconds, and it does not accept the owner
token, a join token, or `X-HavnAI-Token`. Do not loosen that check. A static
`--account-token` dies while the drill is still polling, so the live run reads
a fresh session before every account request.

## Credit budget

Account jobs are charged in ledger units:

`units = quoted credit cost × scale × batch_size`

`scale` is 1000 (`server/account_ledger.py`). The account page shows
`available_units / scale`, so the number on that page is the credit cost, not
the raw unit count. Price does not change with steps, dimensions, or duration.
Music is multiplied by the resolved `batch_size`. This drill sends `batch_size: 1`
so it does not inherit the ACE-Step manifest default of 2.

Default models, from the registry pipeline prices and from live
`GET https://api.joinhavn.io/credits/cost` on 2026-11-19:

| Job | Model | Task | Credits each | Units each |
| --- | --- | --- | --- | --- |
| Image | `juggernautxl_ragnarokby` | `IMAGE_GEN` | 1 | 1,000 |
| LTX video | `ltx_video_distilled` | `LTX_VIDEO_GEN` | 5 | 5,000 |
| Music | `ace_step_1_5_turbo` | `MUSIC_GEN` | 1 | 1,000 |

A 30-job drill is 10 image + 10 LTX video + 10 music:

`10×1 + 10×5 + 10×1 = 70 credits = 70,000 units`

Marcus's account page showed 29 credits available, which is 29,000 units.
That does not cover the drill. It is 41 credits (41,000 units) short. The
smallest pack is Starter (`server/account_payments.py`): 50 credits, 50,000
units. One additional Starter purchase through the normal Stripe checkout on
https://joinhavn.io brings 29 + 50 = 79 credits, which covers 70 with 9 left
at these prices.

Quotes are not a reservation or a price lock. `--preflight` and `--execute`
both re-read `/credits/cost` and `/v2/account/credits`. Do not execute while
preflight says the balance is short. Re-check with:

```bash
curl -fsS "https://api.joinhavn.io/credits/cost?model=juggernautxl_ragnarokby&task_type=IMAGE_GEN"
curl -fsS "https://api.joinhavn.io/credits/cost?model=ltx_video_distilled&task_type=LTX_VIDEO_GEN"
curl -fsS "https://api.joinhavn.io/credits/cost?model=ace_step_1_5_turbo&task_type=MUSIC_GEN"
```

## Adult output on /v2

An active signed-in account can submit adult prompts on `POST /v2/jobs`. There
is no age-verification column, no account adult flag, and no server switch that
must be turned on first. The generation control is the per-job boolean
`sfw_mode`:

- `sfw_mode: true` (the drill default) makes image and video jobs merge
  `SFW_NEGATIVE_PROMPT` into the negative prompt in `server/app.py`. That steers
  the worker away from nude and explicit output.
- `sfw_mode: false` (`--no-sfw-mode`) does not merge that negative. Omitted
  `sfw_mode` is the same as false, because the server checks
  `payload.get("sfw_mode") is True`.

`server/account_jobs.py` still classifies the queued job with
`server/adult_content.py`. A match sets `adult_content=1` and
`adult_policy_reason=adult_content_signal`. The job is still queued. The owning
account can read it and its artifact on authenticated `/v2` routes.
`tests/test_account_jobs.py` covers that with
`test_adult_account_job_and_artifact_metadata_persist_without_blocking_private_generation`.
Public discover, marketplace, playlists, gallery, and legacy static routes keep
the result private. Music generation does not read `sfw_mode`; adult music text
is still classified, and public publish rejects it.

Nothing further is required server-side for this account to request adult
output. `--no-sfw-mode` only stops the SFW negative merge. It does not disable
classification or public isolation, and it does not bypass
`docs/adult-content-policy.md`. Content involving minors, or ambiguous youth in
a sexual context, is prohibited. `server/safety.py` enforces that on legacy
`/submit-job` and `/submit-faceswap-job`. This drill does not send those routes
an exception and does not add a bypass.

## 1. Sign in and fund the account

Marcus signs up and signs in at https://joinhavn.io with Clerk, then buys a
credit pack from the pricing page through Stripe Checkout. Leave that Clerk
session signed in until the drill finishes. Signing out revokes it. Closing the
browser is fine while the Clerk session itself stays active.

The session has to be the one opened from `https://joinhavn.io`, so the minted
JWT's `azp` is in `HAVNAI_ACCOUNT_ORIGINS`. The instance session-token template
is what the website already uses, including `aud` when the coordinator expects
`HAVNAI_CLERK_AUDIENCE` (default `havnai-api`). This helper mints that same
session token. It does not call the JWT-template endpoint, because a custom
template can omit `sid` and can live longer than 120 seconds.

## 2. Mint a session the way Clerk does

On the coordinator host, `CLERK_SECRET_KEY` is already the key the API uses.
Keep it in the environment. Do not put it in argv, the shell history of the
drill command, Jira, or this repo.

```bash
export CLERK_SECRET_KEY          # sk_live_... or sk_test_... already on the host
export HAVNAI_DRILL_CLERK_SESSION_ID=sess_...
python3 scripts/clerk_session_token.py >/dev/null
```

Get `sess_...` from the Clerk dashboard for the signed-in user, or from the
`sid` claim of the `__session` cookie. To use the cookie file instead of the
id:

```bash
install -m 600 /dev/null /secure/joinhavn-session.jwt
# paste the __session cookie value into that file, then:
python3 scripts/clerk_session_token.py \
  --browser-token-file /secure/joinhavn-session.jwt >/dev/null
```

The helper prints one JWT. A check that discards stdout is enough. Do not paste
the JWT into chat or tickets. Default lifetime is 60 seconds, and the helper
refuses a lifetime outside 30..120 seconds.

## 3. Prompts file

The file is a JSON list with one object per image job and one per video job, in
plan order. Music slots are not included. For `--count 30` that is 20 objects:
image, video, image, video, skipping music indexes 3, 6, 9, and so on.

```json
[
  {
    "index": 1,
    "type": "image",
    "prompt": "private adult portrait, studio light",
    "negative_prompt": "blurry, watermark",
    "width": 832,
    "height": 1216,
    "steps": 32,
    "model": "juggernautxl_ragnarokby"
  },
  {
    "index": 2,
    "type": "text_to_video",
    "prompt": "private adult portrait, slow camera move",
    "negative_prompt": "blurry, watermark",
    "width": 480,
    "height": 832,
    "steps": 8
  }
]
```

`prompt` is required. `negative_prompt`, `width`, `height`, `steps`, and
`model` are optional. `index` and `type`, when present, must match the planned
job (`image` or `text_to_video`). Image width and height must each be 256..1536
and steps 5..50, which is the range the server will keep rather than clamp.
Video width and height must both be set, multiples of 32, from 256 to 1280, and
steps 1..150. Omitted image dimensions stay at 832×1216 and 32 steps. Omitted
video dimensions stay on the server preset. Music keeps the built-in cinematic
prompt, instrumental style, 30 second duration, and `batch_size` 1.

`--run-id` recovery must reuse the same prompts file, model list, count, and
sfw flag. The server rejects a changed payload for an existing idempotency key.

## 4. Dry-run, preflight, execute

Run from the repository root. `--token-command` is split without a shell, so
the command itself must not rely on shell expansion. The secret and session id
come from the environment the command inherits.

```bash
export CLERK_SECRET_KEY
export HAVNAI_DRILL_CLERK_SESSION_ID=sess_...

python3 scripts/mixed_model_worker_drill.py \
  --count 30 \
  --dry-run \
  --prompts-file /secure/havn-47-prompts.json \
  --no-sfw-mode \
  --token-command "python3 scripts/clerk_session_token.py"

python3 scripts/mixed_model_worker_drill.py \
  --count 30 \
  --preflight \
  --prompts-file /secure/havn-47-prompts.json \
  --no-sfw-mode \
  --token-command "python3 scripts/clerk_session_token.py"

python3 scripts/mixed_model_worker_drill.py \
  --count 30 \
  --execute \
  --prompts-file /secure/havn-47-prompts.json \
  --no-sfw-mode \
  --token-command "python3 scripts/clerk_session_token.py"
```

Dry-run prints the plan, including prompts, dimensions, `sfw_mode`, and
`session_source`, and does not mint a session or submit jobs. Preflight must
print `passed: true` and a `required_units` of 70000 at the prices above before
execute. Execute quotes again and submits nothing when `budget_sufficient` is
false.

Omit `--no-sfw-mode` to keep the default SFW negative. Use a different
`--token-file /secure/havn-47-session.jwt` only when another process rewrites
that file with a new JWT before each request. A file written once expires
within 120 seconds.

`--node-token` is optional and is sent only to `GET /metrics`. It is not an
account session.

After an interruption, resume the same billable requests with the recorded
`--run-id` and the same plan flags. That recovery is not a new acceptance run.
