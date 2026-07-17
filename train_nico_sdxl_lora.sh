#!/usr/bin/env bash
set -Eeuo pipefail

# Run this script from anywhere. It resolves paths relative to sd-scripts.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Edit these values ──────────────────────────────────────────────────────────
MODEL_PATH="/home/user/ComfyUI/models/checkpoints/novaAnimeXL_ilV180.safetensors"
IMAGE_DIR="/home/user/sd-scripts/dataset/nico_lora"
OUTPUT_DIR="/home/user/ComfyUI/models/loras"
OUTPUT_NAME="nico_sdxl_lora_v2"

RESOLUTION=768
BATCH_SIZE=1
EPOCHS=1
REPEATS=1

NETWORK_DIM=16
NETWORK_ALPHA=16
LEARNING_RATE="1e-4"
SEED=42
# ──────────────────────────────────────────────────────────────────────────────

DATASET_CONFIG="$SCRIPT_DIR/dataset_nico.toml"

[[ -f "$MODEL_PATH" ]] || {
  echo "Error: model checkpoint not found: $MODEL_PATH" >&2
  exit 1
}

[[ -d "$IMAGE_DIR" ]] || {
  echo "Error: dataset directory not found: $IMAGE_DIR" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR"

# Text-encoder output caching is incompatible with caption shuffling and
# caption dropout, so shuffle_caption is intentionally disabled.
cat > "$DATASET_CONFIG" <<EOF
[general]
caption_extension = ".txt"
shuffle_caption = false

[[datasets]]
resolution = [$RESOLUTION, $RESOLUTION]
batch_size = $BATCH_SIZE
enable_bucket = true
bucket_no_upscale = true
bucket_reso_steps = 64

  [[datasets.subsets]]
  image_dir = "$IMAGE_DIR"
  num_repeats = $REPEATS
EOF

echo "Model:   $MODEL_PATH"
echo "Dataset: $IMAGE_DIR"
echo "Output:  $OUTPUT_DIR/$OUTPUT_NAME.safetensors"
echo "Config:  $DATASET_CONFIG"

uv run accelerate launch \
  --num_cpu_threads_per_process=2 \
  "$SCRIPT_DIR/sdxl_train_network.py" \
  --pretrained_model_name_or_path="$MODEL_PATH" \
  --dataset_config="$DATASET_CONFIG" \
  --output_dir="$OUTPUT_DIR" \
  --output_name="$OUTPUT_NAME" \
  --save_model_as="safetensors" \
  --mixed_precision="bf16" \
  --save_precision="bf16" \
  --full_bf16 \
  --network_module="networks.lora" \
  --network_dim="$NETWORK_DIM" \
  --network_alpha="$NETWORK_ALPHA" \
  --network_train_unet_only \
  --learning_rate="$LEARNING_RATE" \
  --unet_lr="$LEARNING_RATE" \
  --optimizer_type="AdamW8bit" \
  --lr_scheduler="constant" \
  --max_train_epochs="$EPOCHS" \
  --save_every_n_epochs=1 \
  --cache_latents \
  --cache_latents_to_disk \
  --cache_text_encoder_outputs \
  --cache_text_encoder_outputs_to_disk \
  --sdpa \
  --persistent_data_loader_workers \
  --max_data_loader_n_workers=1 \
  --seed="$SEED"

echo "Training finished."
