# Z-Image-Turbo bench

Standalone timing and VRAM check for `Tongyi-MAI/Z-Image-Turbo` on the HavnAI RTX 3060. It does not start the node and it does not change the worker, the registry, or routing.

Run it with the HavnAI node stopped, so the card is empty. The script treats 12 GB as the planning number and prints the GPU name and total VRAM it actually sees.

## What it runs

Default variants:

- `q8_0` — `unsloth/Z-Image-Turbo-GGUF` file `z-image-turbo-Q8_0.gguf` (Hub file about 7.2 GB)
- `q5_k_m` — `z-image-turbo-Q5_K_M.gguf` (Hub file about 4.6 GB)
- `juggernautxl` — the `juggernautXL_ragnarokBy` checkpoint, 30 steps, guidance 6.0, registry negative prompt

Optional: `--variants bf16,q8_0,q5_k_m,juggernautxl`. `bf16` loads the unquantized Turbo transformer. That file is about 12.3 GB in bf16, so a 12 GB card is expected to run out of memory. It is off by default.

Every variant draws the same 6 prompts at 1024×1024 with fixed seeds: Nova and Rex portraits, a nebula runway, an empty abyss cockpit, a hangar sign that reads `ASTRA`, and the generic HavnAI fallback prompt from `client.py`. Turbo uses `num_inference_steps=9` (the model card's schedule for 8 DiT forwards) and `guidance_scale=0`. The script counts the forwards it actually runs.

The SDXL baseline follows the live worker, not only the sampler name. `client.py` swaps in `DPMSolverMultistepScheduler.from_config(...)` when the registry sampler contains `dpmpp`, and it does not turn on SDE or Karras. The registry's native size is 832×1216; this bench forces 1024×1024 so the comparison is the same canvas. SDXL gets the juggernautXL negative prompt, which does not ban text. Turbo ignores negatives because guidance is 0. `sign_astra` is the on-image text check for both.

## Why this loader

Diffusers 0.43.0 (the current release, 12 Nov 2026) does support Z-Image GGUF, with one limit: `ZImagePipeline.from_pretrained` cannot open a GGUF file. The working call is `ZImageTransformer2DModel.from_single_file` plus `GGUFQuantizationConfig`, and that module is passed into the pipeline. The same release keeps the GGUF compute-dtype fix in the timestep embedder. `ZImagePipeline.model_cpu_offload_seq` is `text_encoder->transformer->vae`.

That offload is required on 12 GB. Hub file sizes, not measured VRAM:

| piece | size |
| --- | ---: |
| Qwen3 text encoder shards | 8.1 GB |
| VAE | 0.17 GB |
| bf16 transformer | 12.3 GB |
| Q8_0 GGUF | 7.2 GB |
| Q5_K_M GGUF | 4.6 GB |

Text encoder plus Q8_0 is already about 15 GB of weights, so both cannot stay on the card. `enable_model_cpu_offload()` keeps the text encoder on the GPU only while the prompt is encoded, then moves it off before the transformer runs. The first denoising step records each module's device in `results.json` under `denoise_step0`. On a real GPU, `text_encoder_off_gpu` should be true and `transformer_on_gpu` should be true. That is the check for the offload guess.

The pinned packages import together. A Linux CPU check with diffusers 0.43.0, transformers 5.23.0, and torch 2.15.0+cpu reported `ZImageTransformer2DModel.from_single_file` and `model_cpu_offload_seq=text_encoder->transformer->vae`. That check did not load weights or run a forward pass.

The old `zimage` commit `98e0128` loaded base `Tongyi-MAI/Z-Image` in bf16 with `pipe.to(device)` and no offload. That is the wrong checkpoint and the wrong memory plan. This script does not use it.

`run_image_generation` in the worker clamps guidance to at least 1.0. That would turn Turbo's required 0 into CFG. Leave this bench outside the worker until that is a deliberate change.

## Setup on the Windows 3060

Use Python 3.11 or 3.12. Use a separate virtualenv. Installing this requirements file into the node environment would upgrade diffusers past `client/requirements-node.txt`.

Command Prompt, from the repo root, with the node stopped:

```bat
py -3.11 -m venv .venv-zimage-bench
.venv-zimage-bench\Scripts\activate.bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
pip install -r tools\bench\requirements.txt
python tools\bench\zimage_turbo_bench.py --check-support
python tools\bench\zimage_turbo_bench.py --dry-run
python tools\bench\zimage_turbo_bench.py --sdxl-checkpoint D:\havnai-storage\models\creator\juggernautXL_ragnarokBy.safetensors --output D:\havnai-bench\zimage
```

PowerShell is the same commands, except activate with:

```powershell
.\.venv-zimage-bench\Scripts\Activate.ps1
```

If that new venv has no CUDA torch (`torch.cuda.is_available()` is false), install a CUDA wheel before the requirements file. Do not install the default CPU wheel. RTX 3060 is Ampere. This index was up on 18 Nov 2026:

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If the node environment already has a newer CUDA build, match that CUDA version instead of downgrading. `torch>=2.6` is what diffusers 0.43.0 asks for. Check the driver with `nvidia-smi` before the run.

The registry path is `/mnt/d/havnai-storage/models/creator/juggernautXL_ragnarokBy.safetensors`. On native Windows the script also looks at `D:\havnai-storage\models\creator\juggernautXL_ragnarokBy.safetensors`. Pass `--sdxl-checkpoint` when the file lives somewhere else. A missing checkpoint is recorded for the SDXL column and the Turbo variants still run.

The first real run downloads the Q8 and Q5 GGUF files plus the Turbo text encoder and VAE (about 20 GB). Put the cache on a large disk if `C:` is tight:

```bat
python tools\bench\zimage_turbo_bench.py --hf-cache D:\hf-cache --sdxl-checkpoint D:\havnai-storage\models\creator\juggernautXL_ragnarokBy.safetensors --output D:\havnai-bench\zimage
```

`--dry-run` does not download anything. `--check-support` only imports libraries.

## What the output means

`<output>\results.json` and `<output>\results.md` hold the table. Images go to `<output>\images\<variant>\<prompt_id>.png`. `<output>\contact_sheet.png` is the side-by-side sheet. Warm repeats use the first prompt and the same seed, and they are not saved again.

For each variant the files record:

- cold load time, after the files are already on disk
- first-image time
- mean seconds per image over the warm repeats (default 10)
- peak `torch.cuda.max_memory_allocated`
- peak `nvidia-smi` memory.used sample
- peak process RSS and system RAM used
- whether an out-of-memory error was caught
- allocated VRAM and `nvidia-smi` memory.used after unload

Unload matches the worker's single slot: remove offload hooks, move modules to CPU, `gc.collect()`, `torch.cuda.empty_cache()`. Compare `after_unload` with `before_load`. Allocated memory should fall back near the baseline if a one-slot cache would evict cleanly. `nvidia-smi` can stay a bit higher because the driver keeps a context.

An OOM on one variant is recorded and the next variant still runs.

## Useful shorter runs

```bat
python tools\bench\zimage_turbo_bench.py --variants q5_k_m --limit-prompts 1 --warm-runs 2 --output D:\havnai-bench\zimage-short
python tools\bench\zimage_turbo_bench.py --smoke
```

`--smoke` is a CPU fake pipeline. It checks reports, the contact sheet, and that an OOM does not abort the next variant. It is not a model result.
