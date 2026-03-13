# Astra Valkyries × HavnAI: Shared Economy Integration Architecture

**Date:** 2026-03-13
**Status:** Architecture Review — No implementation changes proposed yet
**Scope:** Design-only. Current Public Alpha rendering path and features are untouched.

---

## A. Executive Summary

HavnAI is a decentralized AI compute network with a working wallet → HAI → credits → image generation pipeline. Astra Valkyries is a shmup/rhythm hybrid game with a self-contained gacha economy (credits → pulls → outfits → upgrades). Both systems use "credits" as their core internal currency, but they are currently entirely separate — Astra is a pure client-side game with localStorage persistence and zero backend or wallet integration.

The opportunity is clear: Astra can become an **optional game layer** within the HavnAI ecosystem where players earn and spend shared credits, creating a virtuous loop — play games, earn credits, generate images (or vice versa). The architecture must ensure this integration is **additive** (never required for core HavnAI functionality), **bounded** (game earnings can't destabilize the compute economy), and **incrementally adoptable** (no big-bang rewrite).

**Recommendation:** Astra should live as a **separate frontend** consuming shared HavnAI backend economy APIs. The two systems share a wallet identity and a common credit balance, but Astra's game logic, rendering, and assets remain fully independent.

---

## B. Recommended Architecture

### B.1 Deployment Model: Separate Frontend, Shared Backend

```
┌─────────────────────┐     ┌─────────────────────┐
│   havnai-web        │     │   Astra Valkyries    │
│   (Next.js)         │     │   (Vite + React)     │
│                     │     │                      │
│  • Image generator  │     │  • Shmup gameplay    │
│  • Wallet UI        │     │  • Gacha / shop      │
│  • Dashboard        │     │  • Collection        │
│  • Gallery          │     │  • Leaderboards      │
│  • Marketplace      │     │                      │
└────────┬────────────┘     └────────┬─────────────┘
         │                           │
         │  REST / SSE               │  REST (economy APIs only)
         │                           │
         ▼                           ▼
┌──────────────────────────────────────────────────┐
│              havnai-core (Flask)                  │
│                                                  │
│  Existing APIs:                                  │
│  • /credits/balance          (read balance)      │
│  • /credits/deposit          (add credits)       │
│  • /credits/fund-hai         (HAI → credits)     │
│  • /credits/ledger           (transaction log)   │
│  • /wallet/nonce + /verify   (wallet auth)       │
│                                                  │
│  New APIs (future):                              │
│  • /astra/reward             (bounded earn)      │
│  • /astra/spend              (gacha purchase)    │
│  • /astra/leaderboard        (scores)            │
│                                                  │
│  Economy engine:                                 │
│  • credits.py   (balance, deposit, deduct)       │
│  • settlement.py (ledger, job lifecycle)          │
│  • rewards.py   (node reward computation)        │
└──────────────────────────────────────────────────┘
```

### B.2 Why Separate Frontend (Not Embedded)

| Factor | Embedded (route inside havnai-web) | Separate frontend (recommended) |
|--------|-------------------------------------|----------------------------------|
| **Bundle size** | Adds ~200KB+ game assets/logic to main site | Zero impact on main site |
| **Rendering risk** | Canvas game loop could interfere with Next.js SSR/hydration | Complete isolation |
| **Release cadence** | Game updates require redeploying main site | Independent deploy cycles |
| **Tech mismatch** | Astra uses Vite + React 19, havnai-web uses Next.js | No conflict |
| **Focus protection** | Game UI could distract from alpha testers | Clear separation of concerns |
| **Shared identity** | Same wallet via MetaMask, same backend credits | Same wallet, same credits |
| **Cross-linking** | Can link between sites ("Play Astra" button, "Generate Images" link) | Natural navigation |

### B.3 Authentication Model

Both frontends authenticate via the **same MetaMask wallet address**. The havnai-core backend already identifies users by wallet address across all endpoints. Astra simply needs to:

1. Import or reimplement the wallet connection logic (already abstracted in `lib/wallet.ts`)
2. Send the wallet address with economy API calls
3. Use the existing `/wallet/nonce` + signature verification pattern for authenticated requests

No separate login system or user database needed — the wallet IS the identity.

---

## C. Shared Economy Model

### C.1 Token & Currency Roles

| Currency | Role | Where it lives | Who controls supply |
|----------|------|----------------|---------------------|
| **HAI (ERC-20)** | On-chain token. Store of value. Entry point to economy. | Sepolia blockchain | Smart contract (fixed supply) |
| **Credits** | Off-chain utility currency. Powers all consumption. | havnai-core SQLite `credits` table | Backend only (no client minting) |
| **Astra Game Credits** | **Eliminated in integrated model.** Replaced by shared credits. | — | — |

### C.2 Flow Diagram

```
  User's MetaMask Wallet
        │
        ▼
  HAI Token (on-chain)
        │
        │ transfer to treasury
        ▼
  /credits/fund-hai ──────► Credits Balance (off-chain, per-wallet)
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
              Image Gen        Astra Gacha      Astra Shop
              (1-3 credits)    (spend credits)  (future items)

                    ▲                               │
                    │                               │
              Astra Gameplay ◄──────────────────────┘
              (bounded earn)
```

### C.3 Astra Earn Model (Bounded)

Game performance → credits, but with strict server-enforced limits:

| Parameter | Proposed Value | Rationale |
|-----------|---------------|-----------|
| **Max credits per run** | 5-15 credits | A single image gen costs 1-3 credits; a game run should not exceed ~5 images worth |
| **Daily earn cap** | 50 credits/day | Prevents grinding; ~16-50 images/day from gameplay alone |
| **Cooldown between rewards** | 60 seconds minimum | Prevents rapid-fire bot farming |
| **Score threshold** | Grade B or above | Must demonstrate real gameplay, not AFK |
| **Anti-replay** | Server-side run ID + score hash | Prevents replaying the same result |

**Critical: All earn logic lives on the backend.** The client submits a game result; the server validates and decides the reward. The client never calls `deposit_credits` directly.

### C.4 Astra Spend Model

Gacha pulls and upgrades consume shared credits via backend API:

| Action | Current cost (local) | Proposed shared cost | Notes |
|--------|---------------------|---------------------|-------|
| 1-pull | 120 local credits | 10 shared credits | Rebalanced for shared economy |
| 10-pull | 1000 local credits | 80 shared credits | ~17% savings on bulk |
| Outfit upgrade | Free (shards only) | Free (shards only) | No change needed |

The rebalancing is necessary because local Astra credits have no monetary backing, while shared credits cost real HAI. The gacha cost must feel meaningful but not punishing.

### C.5 Existing Backend Support

The havnai-core `credits.py` module already provides all the primitives needed:

- `get_credit_balance(wallet)` — read balance
- `deposit_credits(wallet, amount, reason)` — add credits (for game rewards)
- `deduct_credits(wallet, amount, job_id)` — atomic spend with balance check
- `release_credits(wallet, amount, reason)` — refunds
- `credit_ledger` table — full audit trail with event types including `LEDGER_REWARD`

The `settlement.py` ledger event types already include `LEDGER_REWARD` — game earnings would use the same pattern as node compute rewards.

---

## D. Major Risks

### D.1 Economy Abuse (HIGH)

**Risk:** Bots or scripts farm Astra gameplay to generate unlimited credits, then use those credits for free image generation (which costs real compute).

**Mitigations:**
- Server-enforced earn caps (daily, per-run, cooldown)
- Score validation on backend (minimum thresholds, statistical anomaly detection)
- Rate limiting on `/astra/reward` endpoint
- Wallet-level fraud scoring (flag wallets with suspicious play patterns)
- Game rewards are a **fraction** of compute costs — even at max earn rate, a bot generates less value than it consumes in operator attention

**Residual risk:** Medium. No client-side game can be fully tamper-proof, but bounded server-side rewards limit the blast radius.

### D.2 User Confusion (MEDIUM)

**Risk:** Users don't understand why "credits" in Astra affect their image generation balance, or vice versa.

**Mitigations:**
- Unified credit balance display in both UIs with clear labeling ("HavnAI Credits")
- Transaction history showing both sources (game rewards + HAI purchases)
- Tooltip/onboarding explaining the shared economy
- Separate "Astra" section in credit ledger for transparency

### D.3 Scope Explosion (HIGH)

**Risk:** Building the integration becomes a multi-month effort that derails Public Alpha launch.

**Mitigations:**
- **No implementation until Public Alpha is stable and tested**
- First milestone is deliberately tiny (read-only balance display)
- Each phase is independently shippable
- Astra game logic changes are minimal — mostly adding API calls alongside existing localStorage logic
- Backend changes are additive — new endpoints, no modification of existing ones

### D.4 Architectural Coupling (MEDIUM)

**Risk:** Astra becomes dependent on havnai-core uptime, or havnai-core starts accumulating game-specific logic.

**Mitigations:**
- Astra should implement graceful degradation — if backend is unreachable, fall back to local-only mode
- Game-specific backend logic is isolated in dedicated `/astra/*` endpoints and a separate `astra_rewards.py` module
- The core credits system (`credits.py`) stays generic — Astra is just another "reason" for deposits
- Astra never touches image generation, node management, or settlement internals

### D.5 Rendering Path Interference (LOW but critical if violated)

**Risk:** Integration work accidentally touches the stable rendering pipeline.

**Mitigations:**
- Astra is a separate frontend — physically cannot modify havnai-web rendering code
- Backend Astra endpoints are new routes, not modifications of existing `/submit-job` or `/results` flows
- Code review gate: no PR touching `engines/`, `submit-job`, `results`, or rendering paths

---

## E. Recommended First Integration Milestone

### "Wallet-Aware Astra" — Prove the Connection

**Goal:** Demonstrate that Astra can authenticate with the same wallet and read the shared credit balance. No earning, no spending — just identity bridging.

**Scope:**
1. Astra adds MetaMask wallet connection (reuse `lib/wallet.ts` patterns from havnai-web)
2. Astra calls `GET /credits/balance?wallet=0x...` and displays the shared balance
3. Astra shows a "Generate Images" link to havnai-web
4. havnai-web shows a "Play Astra" link to the Astra frontend
5. Backend: zero changes (all APIs already exist)

**What this proves:**
- Shared identity works across both frontends
- Credit balance is unified and visible
- The deployment model (separate frontends, shared backend) is viable
- No risk to current HavnAI functionality

**Estimated surface area:** ~3-5 files changed in Astra, 1 link added to havnai-web, 0 backend changes.

**What this does NOT do:** No earning, no spending, no game economy changes. Just "look, same wallet, same balance."

---

## F. Future Phased Roadmap

### Phase 0: Public Alpha Focus (NOW)
- No integration work
- Polish onboarding, wallet UX, image generation stability
- Get real testers using the system
- Astra continues independent development

### Phase 1: Wallet-Aware Astra (Post-Alpha Stabilization)
- MetaMask connection in Astra
- Read-only shared balance display
- Cross-linking between sites
- Backend: no changes

### Phase 2: Shared Spending (After Phase 1 validation)
- Astra gacha pulls deduct from shared credit balance via backend API
- New endpoint: `POST /astra/spend` (validates action, calls `deduct_credits`)
- Astra falls back to local credits if wallet not connected
- Dual-mode: connected players use shared credits, anonymous players use local credits

### Phase 3: Bounded Earning (After Phase 2 is stable)
- Game performance → server-validated credit rewards
- New endpoint: `POST /astra/reward` with anti-abuse protections
- Daily caps, score thresholds, cooldowns
- New module: `server/astra_rewards.py` (isolated from core economy)
- Credit ledger entries tagged with `source: "astra"`

### Phase 4: Cross-Promotion & Flywheel (After Phase 3 proves safe)
- "Earn credits in Astra, generate images in HavnAI" marketing
- Astra leaderboard with credit prizes (funded from treasury/promotions)
- Image generation outputs usable as custom Astra card art (marketplace bridge)
- Seasonal events with limited-time earn multipliers

### Phase 5: On-Chain Game Assets (Long-term vision)
- Astra outfits/cards as on-chain assets (NFTs on HavnAI chain)
- Marketplace: trade Astra items for HAI
- Workflow Registry integration: game asset generation as a registered workflow
- This is where the ARCHITECTURE.md vision of "AI as an economy" meets gaming

---

## Appendix: Current System Inventory

### havnai-core (Python/Flask backend)
- **Credits:** `server/credits.py` — balance, deposit, deduct, release, reserve
- **HAI Funding:** `server/hai_funding.py` — on-chain transfer verification, HAI→credits conversion (1:1)
- **Settlement:** `server/settlement.py` — job lifecycle, quality validation, payouts, credit ledger
- **Rewards:** `server/rewards.py` — node compute reward calculation
- **Auth:** Wallet-based (`/wallet/nonce`, signature verification)
- **DB:** SQLite with tables: `credits`, `hai_fundings`, `job_settlement`, `job_attempts`, `node_payouts`, `credit_ledger`
- **Key constant:** `HAI_TO_CREDITS_RATE = 1.0` (1 HAI = 1 credit)

### havnai-web (Next.js frontend)
- **Wallet:** `lib/wallet.ts` — MetaMask connection, provider detection, chain management
- **HAI Token:** `lib/hai-token.ts` — ERC-20 balance reads, transfer-to-treasury
- **API Client:** `lib/api.js` — `apiGet`/`apiPost` wrappers pointing to `/api` proxy
- **Pages:** generator, wallet, library, marketplace, pricing, nodes, analytics
- **Components:** WalletButton, WalletProvider, StatusBox, HavnAIPrompt

### Astra Valkyries (Vite + React 19 game)
- **State:** `src/context/GameContext.tsx` — React Context + localStorage, manages credits/outfits/scores
- **Economy:** `src/lib/gacha.ts` — pull system (120 credits/pull, rarity rates)
- **Game:** Shmup mode (primary) + rhythm mode (legacy)
- **Persistence:** localStorage only (`astra-valkyries-save` key)
- **Backend:** None. Zero API calls. Fully offline.
- **Wallet:** None. No MetaMask. No authentication.
- **Deploy:** Web (Vite), Desktop (Electron), Mobile (Capacitor)
