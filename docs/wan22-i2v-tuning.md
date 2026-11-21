# Wan 2.2 I2V on a 12 GB RTX 3060

The user path runs ComfyUI, which is the recipe measured on the 3060 on
November 20-21. QuantStack Q4_K_M GGUFs, umT5, the Wan 2.1 VAE, and the
LightX2V 4-step LoRA. Euler, shift 5, 2 high-noise steps then 2 low-noise
steps. Portrait is 480×832, or 480×704 when the source is wider. The default
clip is 129 frames at 16 fps (about 8 seconds) in one pass. 81 frames is the
shorter clip. Peak VRAM is 10.7 to 11.2 GB, so the worker unloads Z-Image
and pauses ACE-Step for the job.

`HAI_WAN22_BACKEND=diffusers` keeps the older in-process loader. That path
is not the measured clip. The numbers live in `client/configs/wan22_i2v.json`.

These VRAM and runtime figures are estimates for an RTX 3060 12 GB with model
CPU offload. This repo does not measure them in CI. Treat them as planning
numbers, and watch the first real job.

## What to run

| Setting | Recommended | Why |
| --- | --- | --- |
| Resolution | 480×832 portrait, or 832×480 landscape | Wan-AI's I2V example uses a 480×832 pixel budget. Both edges fit the Wan VAE multiple of 16. |
| 720p (720×1280 or 1280×720) | Off by default | Activations for 81 frames at 720p exceed a comfortable 12 GB budget even when the expert is offloaded. Opt in with `HAI_WAN22_ALLOW_720P=1` or `"allow_720p": true` on the job, and keep the clip to 49 frames. |
| Frames | 129, or 81 | 129 frames at 16 fps is the measured 8 second clip. 81 frames is about 5 seconds and also fits. |
| Frame rate | 16 fps | The measured export is 16 fps. |
| Steps | 4 | LightX2V's 4-step I2V distill, and the ComfyUI template with the LoRA enabled. |
| High / low split | 2 steps high, then 2 steps low | The same templates run the high-noise expert on steps 0–2 and the low-noise expert on steps 2–4. |
| Boundary ratio | 0.625 | Diffusers switches experts at `boundary_ratio * num_train_timesteps`. For the LightX2V list `[1000, 750, 500, 250]` the midpoint between 750 and 500 is 625, so the ratio is 0.625. The stock Wan 2.2 training boundary (0.875 or 0.9) would leave only the first of those four steps on the high expert. |
| Guidance | 1.0 on both experts | The distill is trained at CFG 1. The base 40-step model uses 3.5; that scale fights this LoRA. |
| Flow shift | 5 | ComfyUI's euler step-index workflow uses shift 5 through ModelSamplingSD3. The diffusers fallback still has an explicit sigma list and is not this recipe. |
| Scheduler | euler | Two KSamplerAdvanced passes, steps 0–2 on the high-noise expert and steps 2–4 on the low-noise expert. |
| LoRA strength | 1.0 high, 1.0 low | LightX2V's published config and the Comfy template. The LoRA is attached per expert and not fused. Fusing a LoRA into a GGUF expert fails in diffusers because the quantized weights are not the original shape. |
| Denoise strength | 1.0, always | Wan I2V does not do SD-style img2img. The source image is encoded and concatenated into the latent for every step, so the clip is fully generated under that condition. A job `strength` below 1 is recorded and ignored. |
| Offload | `enable_model_cpu_offload`, VAE slicing and tiling | Same placement rule as LTX-Video 0.9.8: offload when VRAM is under 16 GB. `sequential` and `group` are available via `HAI_WAN22_OFFLOAD` when the installed diffusers build has them. |
| Text encoder | Local Comfy fp8 file, else diffusers umT5 | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` is a Comfy single file. If `from_single_file` rejects it, the node downloads the umT5 in `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (`text_encoder` and `tokenizer`). That file is several gigabytes and stays on CPU under model offload. |

## VRAM and runtime on a 3060

Q4_K_M of a 14B expert is on the order of 8 GB. The two experts together do
not fit in 12 GB, so only one expert is on the GPU during its steps. The umT5
encoder and the VAE are offloaded around it. Activation memory, not the
checkpoint, is what makes 720p fail.

| Profile | Resolution | Frames | Steps | VRAM | Time |
| --- | --- | --- | --- | --- | --- |
| Default, fits | 480×832 | 81 | 4 | About 10–12 GB peak with model CPU offload. The other expert is on CPU. | About 8–20 minutes. Most of that is moving the expert on and off the card. |
| Shorter 480p | 480×832 | 49 | 4 | Similar peak, less activation memory. | About 5–12 minutes. |
| 720p, opt-in | 720×1280 | 49 | 4 | Often out of memory, or slow enough to spill into shared RAM. | Not a supported default. Budget 20–40 minutes if it completes. |
| Base model, no LoRA | 480×832 | 81 | 40, CFG 3.5 | Same weights, many more steps, and CFG doubles the forward. | Several times the 4-step run. Not what this worker does. |

## Motion presets

Named presets live in `client/configs/wan22_i2v.json` under `motion_presets`.
A job sets `motion_preset` to the name. The preset sentence is appended to the
prompt. The shared negative prompt is used when the job does not send its own.

The presets are idle portrait motion for a creator still: hair in a breeze,
a slow head turn, a weight shift, breathing and idle sway, a slow blink and
a soft smile, a slow camera push-in, and fabric movement. They are not
explicit. The negative prompt asks the sampler to avoid jitter, warping, and
face distortion.

| Name | What it adds |
| --- | --- |
| `hair_breeze` | Hair lifts and settles in a soft breeze. |
| `slow_head_turn` | A slow head turn, shoulders relaxed. |
| `weight_shift` | A slow shift of weight. |
| `breathing_sway` | Quiet breathing and a small idle sway. |
| `blink_smile` | A slow blink and a small smile. |
| `camera_push` | A slow camera push-in. |
| `fabric_movement` | Clothing moving in a light breeze. |

## Overrides

| Layer | Example |
| --- | --- |
| Config | `client/configs/wan22_i2v.json`, or `HAI_WAN22_CONFIG` |
| Environment | `HAI_WAN22_STEPS`, `HAI_WAN22_GUIDANCE`, `HAI_WAN22_FRAMES`, `HAI_WAN22_SHIFT`, `HAI_WAN22_LORA_HIGH`, `HAI_WAN22_LORA_LOW`, `HAI_WAN22_OFFLOAD`, `HAI_WAN22_ALLOW_720P`, `HAI_WAN22_ROOT` |
| Job | `steps`, `guidance`, `guidance_2`, `width`, `height`, `frames`, `fps`, `flow_shift`, `boundary_ratio`, `high_noise_steps`, `lora_scale_high`, `lora_scale_low`, `motion_preset`, `negative_prompt`, `allow_720p` |

`HAI_WAN22_I2V=1` is the admission flag. It does not change the numbers above.

## Sources

- Wan-AI model card, `Wan-AI/Wan2.2-I2V-A14B-Diffusers`. The diffusers sample is 480×832, 81 frames, 16 fps, guidance 3.5, 40 steps, for the base model without the distill LoRA. https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers
- Diffusers Wan pipeline reference. `transformer` is the high-noise expert, `transformer_2` is the low-noise expert, and the switch is `boundary_ratio * num_train_timesteps`. https://huggingface.co/docs/diffusers/api/pipelines/wan
- QuantStack GGUF load. `WanTransformer3DModel.from_single_file` with `GGUFQuantizationConfig`, `config=Wan-AI/Wan2.2-I2V-A14B-Diffusers`, `subfolder=transformer` and `transformer_2`, then `enable_model_cpu_offload()`. https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/discussions/5
- LightX2V distill LoRAs. Separate high and low files, strength 1.0, and the 4-step list `[1000, 750, 500, 250]`. https://huggingface.co/lightx2v/Wan2.2-Distill-Loras
- LightX2V shift note. Shift 1.0 is expected when the sigma list is passed explicitly. https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/16
- ComfyUI / LightX2V template settings. 4 steps, high expert on steps 0–2, low expert on steps 2–4, CFG 1, LoRA strength 1.0 on each expert. https://www.genvidkit.com/blog/wan-2-2-lightx2v-lora
- Diffusers issue on GGUF LoRA fusion. Fusion unpacks badly against GGUF uint weights, so this worker never calls `fuse_lora`. https://github.com/huggingface/diffusers/issues/12047
