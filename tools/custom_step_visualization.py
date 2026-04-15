#!/usr/bin/env python3
"""
Custom Input - Per-Diffusion-Step Visualization
================================================

Run step visualization on a single image + text prompt.
No VBVR-Bench data required.

Usage:
    python tools/custom_step_visualization.py \
        --model wan2.2 \
        --image ./my_image.png \
        --prompt "A cat walks across the room" \
        --output_dir ./output/custom_step \
        --num_frames 81
"""

import torch
import json
import os
import argparse
from PIL import Image

from diffsynth.utils.data import save_video


def parse_vis_steps(spec):
    if spec == 'all':
        return None
    steps = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            steps.update(range(int(lo), int(hi) + 1))
        else:
            steps.add(int(part))
    return steps


def make_step_callback(step_output_dir, fps=16):
    os.makedirs(step_output_dir, exist_ok=True)
    def step_callback(step_idx, total_steps, step_video):
        step_path = os.path.join(step_output_dir, f"step_{step_idx:03d}.mp4")
        save_video(step_video, step_path, fps=fps, quality=5)
        print(f"  Saved step {step_idx}/{total_steps}: {step_path}")
    return step_callback


# ─── Pipeline builders ──────────────────────────────────────────────────────

def build_wan22_pipeline(args, vram_config):
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        redirect_common_files=False,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="google/umt5-xxl/"),
    )
    if args.high_noise_lora_path:
        pipe.load_lora(pipe.dit, args.high_noise_lora_path, alpha=args.lora_alpha)
    if args.low_noise_lora_path:
        pipe.load_lora(pipe.dit2, args.low_noise_lora_path, alpha=args.lora_alpha)
    return pipe

def build_wan21_pipeline(args, vram_config):
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        redirect_common_files=False,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="google/umt5-xxl/"),
    )
    if args.lora_path:
        pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)
    return pipe

def build_ltx_pipeline(args, vram_config):
    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="transformer.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="video_vae_decoder.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vae_decoder.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vocoder.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="video_vae_encoder.safetensors", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
    )
    if args.lora_path:
        pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)
    return pipe


def main():
    parser = argparse.ArgumentParser(
        description="Per-diffusion-step visualization with custom image + prompt")

    parser.add_argument("--model", type=str, required=True,
                        choices=["wan2.2", "wan2.1", "ltx2.3"])
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for generation")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=81)

    lora = parser.add_argument_group("LoRA (optional)")
    lora.add_argument("--lora_path", type=str, default=None)
    lora.add_argument("--high_noise_lora_path", type=str, default=None)
    lora.add_argument("--low_noise_lora_path", type=str, default=None)
    lora.add_argument("--lora_alpha", type=float, default=1.0)

    parser.add_argument("--vis_steps", type=str, default="all",
                        help="'all' or ranges like '0-19,45-49'")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--save_noise_schedule", action="store_true")
    args = parser.parse_args()

    vis_steps = parse_vis_steps(args.vis_steps)
    is_ltx = args.model == "ltx2.3"

    vram_config = {
        "offload_dtype": torch.bfloat16, "offload_device": "cpu",
        "onload_dtype": torch.bfloat16, "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }

    input_image = Image.open(args.image)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Custom Step Visualization")
    print(f"  model:    {args.model}")
    print(f"  image:    {args.image} ({input_image.width}x{input_image.height})")
    print(f"  prompt:   {args.prompt[:80]}...")
    print(f"  frames:   {args.num_frames}")
    print(f"  steps:    {args.num_inference_steps}")
    print(f"  vis:      {args.vis_steps}")
    print("=" * 60)

    if args.model == "wan2.2":
        pipe = build_wan22_pipeline(args, vram_config)
    elif args.model == "wan2.1":
        pipe = build_wan21_pipeline(args, vram_config)
    else:
        pipe = build_ltx_pipeline(args, vram_config)

    if args.save_noise_schedule and not is_ltx:
        pipe.scheduler.set_timesteps(args.num_inference_steps, shift=5.0)
        noise_schedule = [
            {"step": i, "timestep": round(ts.item(), 4), "sigma": round(sig.item(), 6)}
            for i, (sig, ts) in enumerate(zip(pipe.scheduler.sigmas, pipe.scheduler.timesteps))
        ]
        sched_path = os.path.join(args.output_dir, "noise_schedule.json")
        with open(sched_path, 'w') as f:
            json.dump(noise_schedule, f, indent=2)
        print(f"Saved noise schedule: {sched_path}")

    step_dir = os.path.join(args.output_dir, "steps")
    callback = make_step_callback(step_dir, fps=args.fps)

    neg = args.negative_prompt or (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
        "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
        "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
        "背景人很多，倒着走"
    ) if not is_ltx else (args.negative_prompt or (
        "blurry, out of focus, overexposed, underexposed, low contrast, "
        "washed out colors, excessive noise, grainy texture, poor lighting, "
        "flickering, motion blur, distorted proportions, artifacts"
    ))

    output_path = os.path.join(args.output_dir, "generated.mp4")

    if is_ltx:
        from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
        video, audio = pipe(
            prompt=args.prompt, negative_prompt=neg,
            input_images=[input_image], input_images_indexes=[0],
            input_images_strength=1.0, num_frames=args.num_frames,
            seed=args.seed, tiled=True,
            height=input_image.height, width=input_image.width,
            num_inference_steps=args.num_inference_steps,
            step_callback=callback, vis_steps=vis_steps,
        )
        write_video_audio_ltx2(video=video, audio=audio, output_path=output_path,
                               fps=args.fps, audio_sample_rate=pipe.audio_vocoder.output_sampling_rate)
    else:
        video = pipe(
            prompt=args.prompt, negative_prompt=neg,
            input_image=input_image, num_frames=args.num_frames,
            seed=args.seed, tiled=True,
            height=input_image.height, width=input_image.width,
            num_inference_steps=args.num_inference_steps,
            step_callback=callback, vis_steps=vis_steps,
        )
        save_video(video, output_path, fps=args.fps, quality=5)

    print(f"\nFinal video: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
