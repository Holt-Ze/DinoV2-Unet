# DINOv2-UNet Pseudocode

This document mirrors the current compact implementation: core training and
evaluation stay in `seg/training.py`, CLI orchestration stays in `seg/pipeline.py`,
and ablation sweeps stay in `run_ablation_studies.py`.

## Algorithm 1: Forward Pass

```text
Input:
  x: RGB batch, shape B x 3 x H x W

Encoder:
  tokens = DINOv2.patch_embed(x)
  tokens = add_position_embedding(tokens)

  features = []
  for block_index, block in DINOv2.blocks:
      tokens = block(tokens)
      if block_index in out_indices:
          feature = remove_cls_token(tokens)
          feature = reshape_tokens_to_grid(feature)
          feature = conv1x1(feature, embed_dim -> 256)
          features.append(feature)

Decoder:
  features = reverse(features)
  y = lateral_conv(features[0])

  for skip in features[1:]:
      y = bilinear_upsample(y, size=skip.spatial_size)
      if decoder_type == "complex":
          skip = attention_gate(gate=y, skip=skip)
      y = concat(y, lateral_conv(skip))
      y = conv_block(y)

Head:
  logits = segmentation_head(y)
  logits = bilinear_upsample(logits, size=(H, W))

Output:
  {"main": logits}
  plus {"aux_0", "aux_1", "aux_2"} during training when deep supervision is on
```

## Algorithm 2: Training

```text
Input:
  TrainConfig, dataset spec

Setup:
  set_seed(seed)
  model = DinoV2UNet(config)
  if img_size is not divisible by patch_size:
      round img_size up to the next patch multiple

  datasets = build train/val/test datasets
  loaders = build DataLoaders
  optimizer = AdamW(
      encoder params at lr_backbone,
      decoder and aux heads at lr
  )
  scheduler = linear warmup + cosine decay
  loss = ComboLoss(main) + aux_weight_scale * auxiliary losses
  early_stopper = EarlyStopping(best.pt)

Loop:
  for epoch in range(epochs):
      train one epoch with AMP, gradient clipping, scheduler step per batch
      val_metrics = evaluate(validation loader)
      score = (val_metrics.mDice + val_metrics.mIoU) / 2
      save metrics_history.json
      update early stopping with score
      if patience is exceeded:
          break

Final:
  load best.pt
  test_metrics = evaluate(test loader, optional horizontal-flip TTA)
  save a small vis_test sample
  return best validation metrics and test metrics
```

## Algorithm 3: Dataset Handling

```text
For each dataset:
  resolve image/mask folder layout from DatasetSpec
  list supported image extensions
  shuffle names with the configured seed

If num_folds <= 1:
  train = first 80%
  val   = next 10%
  test  = final 10%

If num_folds > 1:
  test = fold slice
  val  = first 10% of remaining names
  train = all remaining names

For each sample:
  load image as RGB
  load mask as grayscale
  apply split-specific albumentations transform
  threshold mask to binary tensor
```

## Algorithm 4: Ablation Runner

```text
Input:
  dataset, data_dir, axes, optional values per axis

Generate:
  configs = cartesian_product(axis values)
  optionally sample configs with deterministic seed 42

For each config and seed:
  convert config to train.py arguments
  set run save dir to runs/ablation_runs/run_xxxx
  run train.py as a subprocess
  read metrics_history.json
  record best validation mDice and mIoU

Output:
  ablation_results/summary.csv
  optional heatmaps when --plot-heatmaps and exactly two axes are used
```
