#!/usr/bin/env python3
"""
Token-Level Feature Map Visualization (VBVR-Bench)
===================================================

Auto-processes all splits (In-Domain_50, Out-of-Domain_50) and all task
directories under VBVR-Bench evaluation data.

Usage:
    python tools/token_visualization.py \
        --model wan2.2 \
        --eval_root ./data/VBVR-Bench \
        --output_root ./output/token_viz/wan2.2 \
        --layers 0 10 20 30 39
"""

import torch
import numpy as np
import os
import subprocess
import argparse
from collections import defaultdict
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffsynth.utils.data import save_video

EVAL_SPLITS = ["In-Domain_50", "Out-of-Domain_50"]


# ═══════════════════════════════════════════════════════════════════════════
#  Feature Capture via PyTorch forward hooks
# ═══════════════════════════════════════════════════════════════════════════

class FeatureCapture:
    def __init__(self, models_dict, layer_ids, blocks_attr="blocks"):
        self.models = {k: v for k, v in models_dict.items() if v is not None}
        self.layer_ids = sorted(layer_ids)
        self.blocks_attr = blocks_attr
        self._hooks = []
        self._features = {}
        self._grid_size = None

    def _make_block_hook(self, layer_id):
        def hook_fn(module, inp, out):
            if layer_id not in self._features:
                self._features[layer_id] = out.detach().cpu().float()
        return hook_fn

    def _make_grid_hook(self):
        def hook_fn(module, inp, out):
            if self._grid_size is None:
                self._grid_size = (out.shape[2], out.shape[3], out.shape[4])
        return hook_fn

    def enable(self):
        for _name, model in self.models.items():
            if hasattr(model, 'patch_embedding'):
                h = model.patch_embedding.register_forward_hook(self._make_grid_hook())
                self._hooks.append(h)
            blocks = getattr(model, self.blocks_attr, None)
            if blocks is None:
                continue
            for lid in self.layer_ids:
                if lid < len(blocks):
                    h = blocks[lid].register_forward_hook(self._make_block_hook(lid))
                    self._hooks.append(h)

    def disable(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self):
        self._features.clear()
        self._grid_size = None

    def collect(self):
        return dict(self._features), self._grid_size


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization helpers
# ═══════════════════════════════════════════════════════════════════════════

def reshape_to_spatial(feat, f, h, w):
    B, N, D = feat.shape
    if N != f * h * w:
        feat = feat[:, N - f * h * w:]
    return feat.reshape(B, f, h, w, D)


def viz_channel_norm(feat_sp, save_path, title=""):
    norm = torch.norm(feat_sp[0], dim=-1).numpy()
    n = norm.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4 * n + 1, 4), squeeze=False)
    for i in range(n):
        im = axes[0, i].imshow(norm[i], cmap="viridis", aspect="auto")
        axes[0, i].set_title(f"Frame {i}")
        axes[0, i].axis("off")
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def viz_pca_map(feat_sp, save_path, title=""):
    from sklearn.decomposition import PCA
    feat = feat_sp[0]
    f, h, w, D = feat.shape
    flat = feat.reshape(-1, D).numpy()
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(flat)
    for c in range(3):
        lo, hi = rgb[:, c].min(), rgb[:, c].max()
        rgb[:, c] = (rgb[:, c] - lo) / (hi - lo + 1e-8)
    rgb = rgb.reshape(f, h, w, 3)
    fig, axes = plt.subplots(1, f, figsize=(4 * f + 1, 4), squeeze=False)
    for i in range(f):
        axes[0, i].imshow(rgb[i])
        axes[0, i].set_title(f"Frame {i}")
        axes[0, i].axis("off")
    pct = pca.explained_variance_ratio_.sum() * 100
    plt.suptitle(f"{title}  (var={pct:.1f}%)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def viz_norm_across_layers(all_feats, save_path, title=""):
    lids = sorted(all_feats.keys())
    norms = [torch.norm(all_feats[l][0], dim=-1).mean().item() for l in lids]
    fig, ax = plt.subplots(figsize=(max(6, len(lids) * 0.5), 4))
    ax.bar(range(len(lids)), norms, tick_label=[str(l) for l in lids], color="steelblue")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Token Norm"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()


def viz_layer_grid(all_feats, grid_size, save_path, title=""):
    f, h, w = grid_size
    lids = sorted(all_feats.keys())
    fig, axes = plt.subplots(len(lids), f, figsize=(3 * f, 2.5 * len(lids)), squeeze=False)
    for li, lid in enumerate(lids):
        sp = reshape_to_spatial(all_feats[lid], f, h, w)[0]
        norm = torch.norm(sp, dim=-1).numpy()
        for fi in range(f):
            axes[li, fi].imshow(norm[fi], cmap="viridis", aspect="auto")
            axes[li, fi].set_xticks([]); axes[li, fi].set_yticks([])
            if fi == 0: axes[li, fi].set_ylabel(f"L{lid}", fontsize=9)
            if li == 0: axes[li, fi].set_title(f"Frame {fi}", fontsize=9)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()


def viz_temporal_energy(feat_sp, save_path, title=""):
    feat = feat_sp[0]
    energy = torch.norm(feat.reshape(feat.shape[0], -1), dim=-1).numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(energy)), energy, "o-", lw=2, ms=8, color="coral")
    ax.set_xlabel("Frame Index"); ax.set_ylabel("Energy (L2)"); ax.set_title(title)
    ax.set_xticks(range(len(energy)))
    plt.tight_layout(); plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()


def viz_channel_importance(feat, save_path, top_k=20, title=""):
    var = feat[0].var(dim=0)
    top_k_actual = min(top_k, len(var))
    vals, ids = var.topk(top_k_actual)
    fig, ax = plt.subplots(figsize=(8, max(3, top_k_actual * 0.25)))
    ax.barh(range(top_k_actual), vals.numpy()[::-1], color="teal")
    ax.set_yticks(range(top_k_actual))
    ax.set_yticklabels([str(i.item()) for i in ids.flip(0)], fontsize=7)
    ax.set_xlabel("Variance"); ax.set_ylabel("Channel ID"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()


def viz_top_channels(feat_sp, save_path, top_k=6, title=""):
    feat = feat_sp[0]
    f, h, w, D = feat.shape
    var = feat.reshape(-1, D).var(dim=0)
    top_k_actual = min(top_k, D)
    ids = var.topk(top_k_actual).indices
    fig, axes = plt.subplots(top_k_actual, f, figsize=(3 * f, 2.5 * top_k_actual), squeeze=False)
    for ki, ch in enumerate(ids):
        vmin, vmax = feat[:, :, :, ch].min().item(), feat[:, :, :, ch].max().item()
        for fi in range(f):
            axes[ki, fi].imshow(feat[fi, :, :, ch].numpy(), cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
            axes[ki, fi].set_xticks([]); axes[ki, fi].set_yticks([])
            if fi == 0: axes[ki, fi].set_ylabel(f"Ch{ch.item()}", fontsize=8)
            if ki == 0: axes[ki, fi].set_title(f"Fr{fi}", fontsize=9)
    plt.suptitle(title, fontsize=11)
    plt.tight_layout(); plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()


def _safe_viz(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"    Warning: {fn.__name__} failed: {e}")


def generate_step_visualizations(all_feats, grid_size, step_dir, step_id, layer_ids):
    f, h, w = grid_size
    os.makedirs(step_dir, exist_ok=True)
    for lid in layer_ids:
        if lid not in all_feats:
            continue
        feat = all_feats[lid]
        sp = reshape_to_spatial(feat, f, h, w)
        pfx = os.path.join(step_dir, f"L{lid:02d}")
        _safe_viz(viz_channel_norm, sp, f"{pfx}_norm.png", f"Step {step_id} L{lid} Norm")
        _safe_viz(viz_pca_map, sp, f"{pfx}_pca.png", f"Step {step_id} L{lid} PCA")
        _safe_viz(viz_temporal_energy, sp, f"{pfx}_temporal.png", f"Step {step_id} L{lid} Temporal Energy")
        _safe_viz(viz_channel_importance, feat, f"{pfx}_ch_imp.png", title=f"Step {step_id} L{lid} Ch Importance")
        _safe_viz(viz_top_channels, sp, f"{pfx}_top_ch.png", title=f"Step {step_id} L{lid} Top Channels")
    _safe_viz(viz_norm_across_layers, all_feats,
              os.path.join(step_dir, "norm_across_layers.png"), title=f"Step {step_id} - Norm Across Layers")
    _safe_viz(viz_layer_grid, all_feats, grid_size,
              os.path.join(step_dir, "layer_grid.png"), title=f"Step {step_id} - Layer Grid")


class StepSummaryTracker:
    def __init__(self):
        self.step_norms = defaultdict(list)
        self.step_temporal_energy = defaultdict(list)
        self.step_ids = []

    def add(self, step_id, all_feats, grid_size):
        self.step_ids.append(step_id)
        f, h, w = grid_size
        for lid, feat in all_feats.items():
            self.step_norms[lid].append(torch.norm(feat[0], dim=-1).mean().item())
            sp = reshape_to_spatial(feat, f, h, w)
            energy = torch.norm(sp[0].reshape(f, -1), dim=-1).numpy()
            self.step_temporal_energy[lid].append(energy)

    def plot_summary(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        lids = sorted(self.step_norms.keys())
        if not lids:
            return
        fig, ax = plt.subplots(figsize=(12, 5))
        for lid in lids:
            ax.plot(self.step_ids, self.step_norms[lid], "o-", label=f"L{lid}", ms=3)
        ax.set_xlabel("Diffusion Step"); ax.set_ylabel("Mean Token Norm")
        ax.set_title("Feature Norm Evolution Across Steps"); ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "norm_evolution.png"), dpi=120, bbox_inches="tight")
        plt.close()
        for lid in lids:
            energies = np.array(self.step_temporal_energy[lid])
            fig, ax = plt.subplots(figsize=(max(6, energies.shape[1] * 1.5), max(4, len(self.step_ids) * 0.15)))
            im = ax.imshow(energies, aspect="auto", cmap="hot")
            ax.set_xlabel("Frame Index"); ax.set_ylabel("Step")
            ax.set_yticks(range(len(self.step_ids))); ax.set_yticklabels(self.step_ids, fontsize=6)
            ax.set_title(f"Layer {lid} - Temporal Energy Heatmap"); plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"temporal_heatmap_L{lid:02d}.png"), dpi=120, bbox_inches="tight")
            plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════════════

def get_video_frame_count(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-count_packets", "-show_entries", "stream=nb_read_packets",
           "-of", "csv=p=0", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(result.stdout.strip())


def parse_viz_steps(spec):
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


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline builders (same as step_visualization.py)
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Token-level feature map visualization on VBVR-Bench")

    parser.add_argument("--model", type=str, required=True,
                        choices=["wan2.2", "wan2.1", "ltx2.3"])

    lora = parser.add_argument_group("LoRA (optional)")
    lora.add_argument("--lora_path", type=str, default=None)
    lora.add_argument("--high_noise_lora_path", type=str, default=None)
    lora.add_argument("--low_noise_lora_path", type=str, default=None)
    lora.add_argument("--lora_alpha", type=float, default=1.0)

    parser.add_argument("--eval_root", type=str, default="./data/VBVR-Bench")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--layers", type=int, nargs="+", default=[0, 10, 20, 30, 39])
    parser.add_argument("--viz_steps", type=str, default="all")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--save_step_videos", action="store_true")
    args = parser.parse_args()

    viz_step_set = parse_viz_steps(args.viz_steps)
    is_ltx = args.model == "ltx2.3"

    if is_ltx:
        print("Note: LTX2.3 spatial heatmaps are experimental.")

    vram_config = {
        "offload_dtype": torch.bfloat16, "offload_device": "cpu",
        "onload_dtype": torch.bfloat16, "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }

    print("=" * 60)
    print("Token-Level Feature Map Visualization (VBVR-Bench)")
    print(f"  model:       {args.model}")
    print(f"  eval_root:   {args.eval_root}")
    print(f"  output_root: {args.output_root}")
    print(f"  layers:      {args.layers}")
    print(f"  viz_steps:   {args.viz_steps}")
    print("=" * 60)

    if args.model == "wan2.2":
        pipe = build_wan22_pipeline(args, vram_config)
        blocks_attr = "blocks"
    elif args.model == "wan2.1":
        pipe = build_wan21_pipeline(args, vram_config)
        blocks_attr = "blocks"
    else:
        pipe = build_ltx_pipeline(args, vram_config)
        blocks_attr = "transformer_blocks"

    models_dict = {"dit": pipe.dit}
    if hasattr(pipe, 'dit2') and pipe.dit2 is not None:
        models_dict["dit2"] = pipe.dit2

    fc = FeatureCapture(models_dict, layer_ids=args.layers, blocks_attr=blocks_attr)
    fc.enable()

    neg_prompt = args.negative_prompt or (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
        "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
        "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
        "背景人很多，倒着走"
    )

    os.makedirs(args.output_root, exist_ok=True)

    for split in EVAL_SPLITS:
        split_path = os.path.join(args.eval_root, split)
        split_output = os.path.join(args.output_root, split)

        if not os.path.isdir(split_path):
            print(f"Split directory {split_path} does not exist, skipping")
            continue

        task_dirs = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        print(f"\n{'='*60}")
        print(f"Split: {split} ({len(task_dirs)} tasks)")
        print(f"{'='*60}")

        for task_dir in task_dirs:
            task_path = os.path.join(split_path, task_dir)
            task_output = os.path.join(split_output, task_dir)

            sample_dirs = sorted([d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))])
            if args.max_samples:
                sample_dirs = sample_dirs[:args.max_samples]

            print(f"\nTask: {task_dir} ({len(sample_dirs)} samples)")

            for sample_dir in sample_dirs:
                sample_path = os.path.join(task_path, sample_dir)
                first_frame = os.path.join(sample_path, "first_frame.png")
                gt_video = os.path.join(sample_path, "ground_truth.mp4")
                prompt_file = os.path.join(sample_path, "prompt.txt")

                out_dir = os.path.join(task_output, sample_dir)
                out_video = os.path.join(out_dir, "generated.mp4")

                if os.path.exists(out_video):
                    print(f"  Skip (exists): {sample_dir}")
                    continue
                if not all(os.path.exists(p) for p in [first_frame, gt_video, prompt_file]):
                    print(f"  Skip (missing files): {sample_dir}")
                    continue

                os.makedirs(out_dir, exist_ok=True)
                input_image = Image.open(first_frame)
                num_frames = get_video_frame_count(gt_video)
                with open(prompt_file) as f:
                    prompt = f.read().strip()

                print(f"  >> {sample_dir}: {num_frames} frames")
                tracker = StepSummaryTracker()

                def make_step_callback(out_dir, tracker, fc, viz_step_set, layer_ids, save_step_videos, fps):
                    def step_callback(step_index, total_steps, step_video):
                        feats, grid_size = fc.collect()
                        if not feats or grid_size is None:
                            fc.reset()
                            return
                        tracker.add(step_index, feats, grid_size)
                        if viz_step_set is None or step_index in viz_step_set:
                            step_dir = os.path.join(out_dir, "feature_maps", f"step_{step_index:04d}")
                            generate_step_visualizations(feats, grid_size, step_dir, step_index, layer_ids)
                        if save_step_videos:
                            sv_dir = os.path.join(out_dir, "step_videos")
                            os.makedirs(sv_dir, exist_ok=True)
                            save_video(step_video, os.path.join(sv_dir, f"step_{step_index:04d}.mp4"), fps=fps, quality=5)
                        fc.reset()
                    return step_callback

                cb = make_step_callback(out_dir, tracker, fc, viz_step_set, args.layers, args.save_step_videos, args.fps)

                try:
                    if is_ltx:
                        from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
                        video, audio = pipe(
                            prompt=prompt, negative_prompt=neg_prompt,
                            input_images=[input_image], input_images_indexes=[0],
                            input_images_strength=1.0, num_frames=num_frames,
                            seed=args.seed, tiled=True,
                            height=input_image.height, width=input_image.width,
                            num_inference_steps=args.num_inference_steps, step_callback=cb,
                        )
                        write_video_audio_ltx2(video=video, audio=audio, output_path=out_video,
                                               fps=args.fps, audio_sample_rate=pipe.audio_vocoder.output_sampling_rate)
                    else:
                        video = pipe(
                            prompt=prompt, negative_prompt=neg_prompt,
                            input_image=input_image, num_frames=num_frames,
                            seed=args.seed, tiled=True,
                            height=input_image.height, width=input_image.width,
                            num_inference_steps=args.num_inference_steps, step_callback=cb,
                        )
                        save_video(video, out_video, fps=args.fps, quality=5)

                    summary_dir = os.path.join(out_dir, "feature_maps", "summary")
                    tracker.plot_summary(summary_dir)
                    print(f"    Saved: {out_video}")

                except Exception as e:
                    print(f"    Error: {e}")
                    import traceback
                    traceback.print_exc()

    fc.disable()
    print("\nAll done!")


if __name__ == "__main__":
    main()
