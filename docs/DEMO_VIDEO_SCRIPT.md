# Demo Video — Shot List and Script

A 90-second cut to open the talk or loop behind the stand. Everything here is
capture-and-assemble: screen recordings you already have the software for, plus
optional generated B-roll produced by the grid itself.

Nothing in this file needs a video editor beyond cutting clips to length and
laying one voiceover track over the top.

---

## Before you record

Bring the stack up exactly as in `DEMO_RUNBOOK.md` §1, then let the heartbeat
loop run for **two full minutes** before you hit record. The dashboard's GPU
utilization values only start varying after a few beats, and a still number
reads as a mockup.

Record at **1920×1080**, hide your bookmarks bar, and put the terminal in a dark
theme so it cuts cleanly against the dashboard.

---

## The cut

| # | Duration | On screen | Capture |
|---|---|---|---|
| 1 | 0:00–0:06 | Astra Valkyries title screen, "PRESS START" pulsing | Screen record `localhost:5173` |
| 2 | 0:06–0:14 | A combat run — 6 seconds of actual play | Screen record `/shmup` |
| 3 | 0:14–0:22 | Run results, grade landing, HAI reward counting up | Screen record `/shmup-results` |
| 4 | 0:22–0:32 | Cut to the grid dashboard. Node count, utilization ticking | Screen record `localhost:5001/dashboard` |
| 5 | 0:32–0:42 | Terminal: `curl -s localhost:5001/jobs/recent \| jq '.jobs[0]'` | Screen record terminal |
| 6 | 0:42–0:56 | Generated B-roll — 3 or 4 outputs, 3s each, hard cuts | See below |
| 7 | 0:56–1:06 | `registry.json` scrolling slowly through the model list | Screen record editor |
| 8 | 1:06–1:16 | Split: dashboard left, game leaderboard right | Two recordings side by side |
| 9 | 1:16–1:30 | Black. White monospace type, one line at a time | Title cards |

---

## Voiceover

Timed to the table above. Roughly 150 words, which lands at a calm pace in 90
seconds — do not rush it.

> **(0:00)** This is a game. It runs in a browser, on a desktop, on a phone.
>
> **(0:14)** When a run ends, the score becomes credits. Not points — credits,
> on a network.
>
> **(0:22)** Because behind it there's a grid. Consumer GPUs, in people's homes,
> pulling jobs off a queue.
>
> **(0:32)** Every job gets a model picked by weighted lottery, a price computed
> from what the compute actually cost, and a payout settled to a wallet.
>
> **(0:42)** The models are the ones most platforms won't host. The filter in
> front of them is the reason we can.
>
> **(1:06)** One coordinator. Twenty-five models. Four GPU classes. And a game
> that spends what the grid earns.
>
> **(1:16)** HavnAI. Private alpha.

---

## Shot 6 — generated B-roll

Run these through the grid so the footage is genuinely your own output. Submit
with `model="auto"` to exercise the router, or name a checkpoint for a specific
look.

```bash
curl -X POST localhost:5001/submit-job \
  -H "Content-Type: application/json" \
  -d '{"wallet":"0x71c7656ec7ab88b098defb751b7401b5f6d8976f",
       "model":"auto",
       "prompt":"<prompt below>",
       "negative_prompt":"lowres, blurry, watermark, text, bad anatomy, extra fingers"}'
```

Prompts chosen to look expensive on a projector — high contrast, strong rim
light, shallow depth of field. Swap them for whatever suits the room; these are
staging-safe defaults that still demonstrate range.

1. `cinematic portrait, rim lighting, wet asphalt reflections, neon signage bokeh, shallow depth of field, 85mm, film grain`
2. `chrome and glass architecture at dusk, long exposure light trails, volumetric fog, ultra wide angle`
3. `studio product shot, single hard key light, deep black background, specular highlights, macro detail`
4. `desert highway at golden hour, heat shimmer, anamorphic lens flare, motion blur`

For a moving shot, submit to the video pipeline instead:

```bash
curl -X POST localhost:5001/submit-job \
  -H "Content-Type: application/json" \
  -d '{"wallet":"0x71c7656ec7ab88b098defb751b7401b5f6d8976f",
       "model":"animatediff",
       "prompt":"slow push through a neon-lit corridor, volumetric haze, cinematic",
       "frames":24,"fps":8,"motion":"zoom-in","base_model":"realisticVision",
       "width":512,"height":512,"scheduler":"DDIM"}'
```

Poll `GET /result/<job_id>` for the MP4 path. On a 12GB card this lands in
roughly 40–70 seconds per clip, so budget for it — generate the night before,
not in the venue.

---

## Shot 9 — closing cards

White monospace on black, one line per card, roughly 3 seconds each. No
animation beyond a hard cut.

```
It routes.
It prices.
It settles.
joinhavn.io
```

---

## If you have no time to edit

Shots 1, 2, 3, 4 in that order, no voiceover, no B-roll. Thirty-two seconds,
cuts on the beat, and it still tells the whole story: a game, a run, a payout, a
grid. Everything else is decoration on top of that spine.
