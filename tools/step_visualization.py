#!/usr/bin/env python3
"""
Per-Diffusion-Step Visualization (VBVR-Bench)
==============================================

Auto-processes all splits (In-Domain_50, Out-of-Domain_50) and all task
directories under VBVR-Bench evaluation data.

Usage:
    python tools/step_visualization.py \
        --model wan2.2 \
        --eval_root ./data/VBVR-Bench \
        --output_root ./output/step_viz/wan2.2
"""

import torch
import json
import os
import subprocess
import argparse
from PIL import Image

from diffsynth.utils.data import save_video

EVAL_SPLITS = ["In-Domain_50", "Out-of-Domain_50"]


def get_video_frame_count(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_packets', '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(result.stdout.strip())


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
        print(f"    Saved intermediate step {step_idx}/{total_steps}: {step_path}")
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
    if args.high_noise_lora_path is not None:
        print(f"Loading high-noise LoRA: {args.high_noise_lora_path}")
        pipe.load_lora(pipe.dit, args.high_noise_lora_path, alpha=args.lora_alpha)
    if args.low_noise_lora_path is not None:
        print(f"Loading low-noise LoRA: {args.low_noise_lora_path}")
        pipe.load_lora(pipe.dit2, args.low_noise_lora_path, alpha=args.lora_alpha)
    if args.high_noise_lora_path is None and args.low_noise_lora_path is None:
        print("Running base model (no LoRA)")
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
    if args.lora_path is not None:
        print(f"Loading LoRA: {args.lora_path}")
        pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)
    else:
        print("Running base model (no LoRA)")
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
    if args.lora_path is not None:
        print(f"Loading LoRA: {args.lora_path}")
        pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)
    else:
        print("Running base model (no LoRA)")
    return pipe


# ─── Inference runners ───────────────────────────────────────────────────────

def run_wan(pipe, args, input_image, prompt, num_frames, callback, vis_steps):
    negative_prompt = args.negative_prompt or (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
        "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
        "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
        "背景人很多，倒着走"
    )
    video = pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        input_image=input_image, num_frames=num_frames,
        seed=args.seed, tiled=True,
        height=input_image.height, width=input_image.width,
        num_inference_steps=args.num_inference_steps,
        step_callback=callback, vis_steps=vis_steps,
    )
    return video, None


def run_ltx(pipe, args, input_image, prompt, num_frames, callback, vis_steps):
    negative_prompt = args.negative_prompt or (
        "blurry, out of focus, overexposed, underexposed, low contrast, "
        "washed out colors, excessive noise, grainy texture, poor lighting, "
        "flickering, motion blur, distorted proportions, artifacts, "
        "cartoonish rendering, 3D CGI look, unrealistic materials"
    )
    video, audio = pipe(
        prompt=prompt, negative_prompt=negative_prompt,
        input_images=[input_image], input_images_indexes=[0],
        input_images_strength=1.0, num_frames=num_frames,
        seed=args.seed, tiled=True,
        height=input_image.height, width=input_image.width,
        num_inference_steps=args.num_inference_steps,
        step_callback=callback, vis_steps=vis_steps,
    )
    return video, audio


# ─── Sample processing ───────────────────────────────────────────────────────

def process_sample(pipe, run_fn, args, is_ltx, vis_steps,
                   sample_path, sample_dir, task_output_dir):
    first_frame_path = os.path.join(sample_path, "first_frame.png")
    ground_truth_path = os.path.join(sample_path, "ground_truth.mp4")
    prompt_path = os.path.join(sample_path, "prompt.txt")
    output_video_path = os.path.join(task_output_dir, f"{sample_dir}.mp4")

    if os.path.exists(output_video_path):
        print(f"  Skipping {sample_dir} - already exists")
        return

    if not all(os.path.exists(p) for p in [first_frame_path, ground_truth_path, prompt_path]):
        print(f"  Skipping {sample_dir} - missing required files")
        return

    input_image = Image.open(first_frame_path)
    num_frames = get_video_frame_count(ground_truth_path)
    with open(prompt_path) as f:
        prompt = f.read().strip()

    print(f"  Processing {sample_dir}: {num_frames} frames")

    if args.save_noise_schedule and not is_ltx:
        pipe.scheduler.set_timesteps(args.num_inference_steps, shift=5.0)
        noise_schedule = []
        for step_idx, (sigma, ts) in enumerate(zip(
                pipe.scheduler.sigmas, pipe.scheduler.timesteps)):
            noise_schedule.append({
                "step": step_idx,
                "timestep": round(ts.item(), 4),
                "sigma": round(sigma.item(), 6),
            })
        schedule_path = os.path.join(task_output_dir, f"{sample_dir}_noise_schedule.json")
        with open(schedule_path, 'w') as nf:
            json.dump(noise_schedule, nf, indent=2)
        print(f"    Saved noise schedule: {schedule_path}")

    step_output_dir = os.path.join(task_output_dir, f"{sample_dir}_steps")
    callback = make_step_callback(step_output_dir, fps=args.fps)

    video, audio = run_fn(pipe, args, input_image, prompt,
                          num_frames, callback, vis_steps)

    if is_ltx and audio is not None:
        from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
        write_video_audio_ltx2(
            video=video, audio=audio,
            output_path=output_video_path, fps=args.fps,
            audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
        )
    else:
        save_video(video, output_video_path, fps=args.fps, quality=5)

    print(f"    Saved: {output_video_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-diffusion-step visualization on VBVR-Bench")

    parser.add_argument("--model", type=str, required=True,
                        choices=["wan2.2", "wan2.1", "ltx2.3"])

    lora = parser.add_argument_group("LoRA (optional)")
    lora.add_argument("--lora_path", type=str, default=None,
                      help="LoRA path (wan2.1 / ltx2.3)")
    lora.add_argument("--high_noise_lora_path", type=str, default=None,
                      help="High-noise DiT LoRA (wan2.2)")
    lora.add_argument("--low_noise_lora_path", type=str, default=None,
                      help="Low-noise DiT LoRA (wan2.2)")
    lora.add_argument("--lora_alpha", type=float, default=1.0)

    parser.add_argument("--eval_root", type=str, default="./data/VBVR-Bench",
                        help="Root of VBVR-Bench data")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--vis_steps", type=str, default="all",
                        help="'all' or ranges like '0-19,45-49'")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--save_noise_schedule", action="store_true")
    args = parser.parse_args()

    vis_steps = parse_vis_steps(args.vis_steps)

    vram_config = {
        "offload_dtype": torch.bfloat16, "offload_device": "cpu",
        "onload_dtype": torch.bfloat16, "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }

    print("=" * 60)
    print("Per-Diffusion-Step Visualization (VBVR-Bench)")
    print(f"  model:       {args.model}")
    print(f"  eval_root:   {args.eval_root}")
    print(f"  output_root: {args.output_root}")
    print(f"  vis_steps:   {args.vis_steps}")
    print("=" * 60)

    is_ltx = args.model == "ltx2.3"
    if args.model == "wan2.2":
        pipe = build_wan22_pipeline(args, vram_config)
        run_fn = run_wan
    elif args.model == "wan2.1":
        pipe = build_wan21_pipeline(args, vram_config)
        run_fn = run_wan
    else:
        pipe = build_ltx_pipeline(args, vram_config)
        run_fn = run_ltx

    os.makedirs(args.output_root, exist_ok=True)

    for split in EVAL_SPLITS:
        split_path = os.path.join(args.eval_root, split)
        split_output_dir = os.path.join(args.output_root, split)

        if not os.path.isdir(split_path):
            print(f"Split directory {split_path} does not exist, skipping")
            continue

        task_dirs = sorted([
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        ])
        print(f"\n{'='*60}")
        print(f"Split: {split} ({len(task_dirs)} tasks)")
        print(f"{'='*60}")

        for task_dir in task_dirs:
            task_path = os.path.join(split_path, task_dir)
            task_output_dir = os.path.join(split_output_dir, task_dir)
            os.makedirs(task_output_dir, exist_ok=True)

            sample_dirs = sorted([
                d for d in os.listdir(task_path)
                if os.path.isdir(os.path.join(task_path, d))
            ])
            if args.max_samples:
                sample_dirs = sample_dirs[:args.max_samples]

            print(f"\nTask: {task_dir} ({len(sample_dirs)} samples)")

            for sample_dir in sample_dirs:
                sample_path = os.path.join(task_path, sample_dir)
                try:
                    process_sample(pipe, run_fn, args, is_ltx, vis_steps,
                                   sample_path, sample_dir, task_output_dir)
                except Exception as e:
                    print(f"  Error on {sample_dir}: {e}")
                    import traceback
                    traceback.print_exc()

        print(f"\nDone split: {split}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
