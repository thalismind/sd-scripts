# Weight Decay Scheduler for LoRA Training

This document describes the dynamic weight decay scheduler feature implemented in `sd-scripts` for LoRA training.

## Overview

The Weight Decay Scheduler provides a way to dynamically adjust weight decay during training, starting with a higher value for early generalization and gradually reducing it for late precision. This approach is supported by theory and best practices in large model fine-tuning.

## Features

- **Linear and Cosine Decay**: Choose between linear interpolation or smooth cosine interpolation
- **Warmup Support**: Optional warmup period with configurable weight decay value
- **Logging Integration**: Automatic logging of current weight decay values
- **Metadata Storage**: Weight decay schedule information is saved in model metadata

## Usage

### Basic Usage

Enable the weight decay scheduler with default settings:

```bash
python train_network.py \
  --weight_decay_schedule \
  --weight_decay_start 0.1 \
  --weight_decay_end 0.01 \
  --weight_decay_mode linear \
  # ... other training arguments
```

### Advanced Usage

Configure warmup and custom decay values:

```bash
python train_network.py \
  --weight_decay_schedule \
  --weight_decay_start 0.2 \
  --weight_decay_end 0.02 \
  --weight_decay_mode cosine \
  --weight_decay_warmup_steps 100 \
  --weight_decay_warmup_value 0.0 \
  # ... other training arguments
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--weight_decay_schedule` | flag | False | Enable dynamic weight decay scheduling |
| `--weight_decay_start` | float | 0.1 | Initial weight decay value |
| `--weight_decay_end` | float | 0.01 | Final weight decay value |
| `--weight_decay_mode` | str | "linear" | Schedule mode: "linear" or "cosine" |
| `--weight_decay_warmup_steps` | int | 0 | Number of warmup steps |
| `--weight_decay_warmup_value` | float | 0.0 | Weight decay value during warmup |

## Recommended Settings

### Style LoRA
```bash
--weight_decay_schedule \
--weight_decay_start 0.1 \
--weight_decay_end 0.01 \
--weight_decay_mode linear
```

### Identity/Portrait LoRA
```bash
--weight_decay_schedule \
--weight_decay_start 0.05 \
--weight_decay_end 0.0 \
--weight_decay_mode cosine
```

### Concept LoRA
```bash
--weight_decay_schedule \
--weight_decay_start 0.2 \
--weight_decay_end 0.02 \
--weight_decay_mode linear
```

## How It Works

1. **Initialization**: The scheduler sets the initial weight decay to the warmup value (if specified) or 0.0
2. **Warmup Phase**: During warmup steps, weight decay remains at the warmup value
3. **Decay Phase**: After warmup, weight decay gradually transitions from start to end value
4. **Logging**: Current weight decay values are logged every 100 steps
5. **Metadata**: Schedule information is saved in the model metadata

## Decay Modes

### Linear Mode
Weight decay decreases linearly from start to end value:
```
weight_decay = start + (end - start) * (step / max_steps)
```

### Cosine Mode
Weight decay decreases smoothly using cosine interpolation:
```
weight_decay = start + (end - start) * 0.5 * (1 + cos(π * (1 - progress)))
```

## Logging

The scheduler automatically logs weight decay values:
- Every 100 steps during training
- In TensorBoard/WandB as `weight_decay/current`
- In model metadata with schedule parameters

## Metadata

The following metadata is saved in the trained model:
- `ss_weight_decay_schedule`: "enabled"
- `ss_weight_decay_start`: Initial weight decay value
- `ss_weight_decay_end`: Final weight decay value
- `ss_weight_decay_mode`: Schedule mode (linear/cosine)
- `ss_weight_decay_warmup_steps`: Number of warmup steps
- `ss_weight_decay_warmup_value`: Warmup weight decay value

## Example Training Command

```bash
python train_network.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="./dataset" \
  --output_dir="./output" \
  --resolution=512 \
  --network_alpha=128 \
  --save_model_as=safetensors \
  --network_module=networks.lora \
  --max_train_epochs=10 \
  --learning_rate=1e-4 \
  --unet_lr=1e-4 \
  --network_dim=128 \
  --batch_size=1 \
  --mixed_precision="fp16" \
  --save_every_n_epochs=1 \
  --weight_decay_schedule \
  --weight_decay_start 0.1 \
  --weight_decay_end 0.01 \
  --weight_decay_mode linear
```

## Technical Details

- The scheduler modifies the optimizer's `weight_decay` parameter directly
- Works with all supported optimizers (AdamW, AdamW8bit, Lion, etc.)
- Compatible with gradient accumulation and multi-GPU training
- No impact on learning rate scheduling
- Safe to use with existing training scripts

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure the `library/weight_decay_scheduler.py` file exists
2. **Invalid Parameters**: Check that warmup_steps < max_train_steps
3. **No Effect**: Verify that `--weight_decay_schedule` is enabled

### Debug Mode

To see detailed weight decay values, check the training logs for:
```
Step X: weight_decay = Y.YYYYYY
```

## Theory

Dynamic weight decay scheduling is based on the principle that:
- **Early training**: Higher weight decay prevents overfitting and encourages generalization
- **Late training**: Lower weight decay allows for fine-tuning and detail preservation

This approach is particularly effective for LoRA training where the goal is to adapt a small set of parameters to new concepts while maintaining the base model's capabilities.