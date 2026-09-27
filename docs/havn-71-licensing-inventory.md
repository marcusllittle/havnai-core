# HAVN-71 Creative Platform Licensing Inventory

Date: 2026-09-27
Core branch: `codex/havn-71-licensing-main`
Core base: current `origin/main` for HAVN-72 launch hardening
Web evidence: current `havnai-web` production branch `feat/havn-11-commercial-accounts`
through merge `82e1e9f`, deployed as `dpl_GnYQ31UvFaE675xmdgdT5aGSFpME`

## Scope

HAVN-71 covers `havnai-core` and `havnai-web` creative-platform models and shipped public assets. HAVN-66 owns the detailed Astra game asset inventory; this file records web exposure of Astra assets as a dependency, not a duplicate approval.

## Core Model Inventory Summary

Source: `server/manifests/registry.json`

- Total enabled model/service entries: 28
- Entries with explicit `source.license` or `license_status`: 6
- Entries missing explicit provenance/license status: 22
- Coordinator-hosted missing-license entries: 22

### Entries With License/Status Metadata

| Entry | Pipeline | Source kind | Current license/status | Release note |
| --- | --- | --- | --- | --- |
| `ltx23_wangp_distilled` | `ltx23_wangp` | `operator` | `research_owner_only`; source license `research / owner only` | Must not be represented as cleared for unrestricted commercial output/use without qualified review or explicit permission. Not redistributed by coordinator. |
| `ltx_video_dev` | `ltx_video` | `hf` | `research_owner_only`; source license references upstream Lightricks repo | Needs upstream terms review before commercial launch claims. |
| `ltx_video_distilled` | `ltx_video` | `hf` | `research_owner_only`; source license references upstream Lightricks repo | Needs upstream terms review before commercial launch claims. |
| `ace_step_1_5_turbo` | `ace_step` | `operator` | `upstream_terms_apply` | Needs ACE-Step upstream license/reference URL and output-use review. |
| `ace_step_1_5_base` | `ace_step` | `operator` | `upstream_terms_apply` | Needs ACE-Step upstream license/reference URL and output-use review. |
| `ace_step_1_5_sft` | `ace_step` | `operator` | `upstream_terms_apply` | Needs ACE-Step upstream license/reference URL and output-use review. |

### Coordinator-Hosted Entries Missing License/Provenance

These entries are enabled in the release manifest and are served by the coordinator or represented as coordinator-hosted artifacts, but the manifest does not record source URL, license, version/source owner, attribution, commercial-use terms, output-use terms, or redistribution terms.

| Entry | Pipeline | Type | File |
| --- | --- | --- | --- |
| `juggernautXL_ragnarokBy` | `sdxl` | checkpoint | `juggernautXL_ragnarokBy.safetensors` |
| `epicrealismXL_vxviiCrystalclear` | `sdxl` | checkpoint | `epicrealismXL_vxviiCrystalclear.safetensors` |
| `babesByStableYogiPony_xlV4` | `sdxl` | checkpoint | `babesByStableYogiPony_xlV4.safetensors` |
| `epicrealismXL_purefix` | `sdxl` | checkpoint | `epicrealismXL_purefix.safetensors` |
| `perfectdeliberate_v60` | `sdxl` | checkpoint | `perfectdeliberate_v60.safetensors` |
| `zavychromaxl_v100` | `sdxl` | checkpoint | `zavychromaxl_v100.safetensors` |
| `perfectdeliberate_v5SD15` | `sd15` | checkpoint | `perfectdeliberate_v5SD15.safetensors` |
| `divineelegancemix_V10` | `sd15` | checkpoint | `divineelegancemix_V10.safetensors` |
| `uberRealisticPornMerge_v23Final` | `sd15` | checkpoint | `uberRealisticPornMerge_v23Final.safetensors` |
| `triomerge_v10` | `sd15` | checkpoint | `triomerge_v10.safetensors` |
| `unstablePornhwa_beta` | `sd15` | checkpoint | `unstablePornhwa_beta.safetensors` |
| `disneyPixarCartoon_v10` | `sd15` | checkpoint | `disneyPixarCartoon_v10.safetensors` |
| `kizukiAnimeHentai_animeHentaiV4` | `sd15` | checkpoint | `kizukiAnimeHentai_animeHentaiV4.safetensors` |
| `aMixIllustrious_aMix` | `sdxl` | checkpoint | `aMixIllustrious_aMix.safetensors` |
| `cyberrealisticPony_v160` | `sdxl` | checkpoint | `cyberrealisticPony_v160.safetensors` |
| `ikastrious_v220` | `sdxl` | checkpoint | `ikastrious_v220.safetensors` |
| `ponyDiffusionV6XL_v615` | `sd15` | checkpoint | `ponyDiffusionV6XL_v615.safetensors` |
| `realisticVisionV60B1_v51HyperVAE` | `sd15` | checkpoint | `realisticVisionV60B1_v51HyperVAE.safetensors` |
| `lyriel_v16` | `sd15` | checkpoint | `lyriel_v16.safetensors` |
| `ltx2` | `ltx2` | diffusers | `Latte-1` |
| `copaxTimeless_xplus2BNSFW1` | `sdxl` | checkpoint | `copaxTimeless_xplus2BNSFW1.safetensors` |
| `animatediff` | `animatediff` | checkpoint | `realisticVisionV60B1_v51HyperVAE.safetensors` |

## Web Asset Inventory Summary

Source checkout: `havnai-web` production branch evidence as of deploy
`dpl_GnYQ31UvFaE675xmdgdT5aGSFpME`

- `public/HavnAI-logo.png`: HavnAI brand asset. Owner/provenance should be recorded by Marcus/HavnAI.
- `public/music-default-cover.png`: shipped fallback cover art used by Discover, playlist, player, and music library surfaces. Owner/provenance/license not recorded in repo.
- `public/create/coastal-light.webp` and `public/create/amber-still-life.webp`: shipped creation-inspiration images. `public/create/README.md` says they are "AI-generated placeholder artwork" and should be replaced with commissioned/brand assets for launch; owner/provenance/license not sufficient for commercial approval.
- `public/astra/**`: 40 shipped Astra images used by public HavnAI pages and SEO. Detailed ownership/licensing should be covered by HAVN-66; HAVN-71 should not close until HAVN-66 evidence covers public web usage or the web references are replaced/disabled.
- Fonts: `_document.tsx` loads Google Fonts `Orbitron` and `Exo 2`; required attribution/notice handling should be confirmed. No local font license notice file is shipped.

## Release Blockers / Required Next Actions

1. Add source owner, exact source URL/reference, version, license, commercial-use terms, output-use terms, redistribution/hosting terms, attribution requirements, and reviewer/owner for every enabled model entry.
2. Treat all 22 coordinator-hosted entries without license metadata as not commercially cleared until documented permission or accepted license evidence exists.
3. For `research_owner_only` and `upstream_terms_apply` entries, route interpretation for qualified review before public commercial claims.
4. Record provenance/owner/license for HavnAI logo, music default cover, and create-page artwork; replace the create placeholders before launch unless documented commercial permission exists.
5. Reuse HAVN-66 for Astra assets, but verify it explicitly covers the 40 public web files under `public/astra/**`.
6. Confirm required public notices/attributions, including model notices and Google Font terms, are shipped or not required.

## Current Recommendation

Do not mark HAVN-71 complete and do not declare HAVN-72 launch-ready. The current codebase contains enabled model and public-asset surfaces with incomplete license/provenance records.
