# HAVN-42 Cloudflare Cache Purge Runbook

Date: 2026-09-27

This runbook covers the current HAVN-14/HAVN-42 launch blocker where origin
private-media behavior is fixed but one exact Cloudflare-cached URL still serves
a private artifact publicly.

## Current Blocker

- Zone: `joinhavn.io`
- Cloudflare zone ID: `68cf9ca63f39dc0550abe73f30e9cd1c`
- Stale URL:
  `https://joinhavn.io/api/static/outputs/audio/job-e634fd8d8f32.mp3`
- Latest observed stale response, 2026-09-27 11:27 UTC:
  - `HTTP/2 200`
  - `cf-cache-status: HIT`
  - `content-type: audio/mpeg`
  - `content-length: 960813`
  - `cache-control: max-age=14400`
  - `age: 9600`
- Cache-busted origin check at the same time returned the correct behavior:
  - `HTTP/2 404`
  - `cache-control: private, no-store`
  - `cf-cache-status: BYPASS`
  - `vary: Authorization`
  - body `{"error":"artifact_not_found"}`

## Required Cloudflare Access

The available Codex Cloudflare connector can read the zone, but cannot purge
cache. Its exact-file purge attempt failed with Cloudflare API error `10000:
Authentication error`.

Use a Cloudflare token/user with permission to purge zone cache for `joinhavn.io`.

## Purge Command

Using a purge-capable Cloudflare token:

```bash
ZONE_ID=68cf9ca63f39dc0550abe73f30e9cd1c
URL='https://joinhavn.io/api/static/outputs/audio/job-e634fd8d8f32.mp3'

curl -fsS "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/purge_cache" \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data "{\"files\":[\"$URL\"]}"
```

Expected purge response includes `"success": true`.

## Required Recheck

After purge, recheck the exact URL without query parameters:

```bash
curl -sS -D - -o /tmp/havn-cache-check-body \
  https://joinhavn.io/api/static/outputs/audio/job-e634fd8d8f32.mp3
cat /tmp/havn-cache-check-body
```

HAVN-14/HAVN-42 can only advance when the exact URL no longer serves the cached
audio body. Passing evidence should show private/not-found behavior similar to:

- `HTTP/2 404` or another non-200 private denial
- `cache-control: private, no-store`
- `cf-cache-status` not serving the stale audio object
- body does not contain the audio artifact bytes

Attach the purge response and exact URL recheck headers/body summary to
HAVN-42, HAVN-14, and HAVN-72.
