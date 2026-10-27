# Evaluation protocol and compatibility

This documents the released evaluation, validation and sampling entry points,
including their metric definitions, saved predictions and compatibility rules.

## Data and spatial transformations

`create_dataset(..., training=False)` is the default. Training opts in explicitly;
evaluation and visualization disable augmentation independently of the split name.
The existing scientific augmentation policy during training is unchanged.

The dataset records `spatial_mode`: signed-capped-asinh inputs use centre padding;
the z-score fallback uses bilinear resizing. All production native-size conversion
calls supply that mode. Padding is removed by cropping the same offsets used on
input; resized tensors are interpolated to the native dimensions with
`align_corners=False`. Interpolation is lossy. Adapters returning native predictions
must actually match the target dimensions; the evaluator no longer guesses a
transform from a size mismatch.

The configured `clip_x0` flag applies only to bounded signed-capped-asinh data.
Z-score predictions remain unbounded even when an older configuration has
`clip_x0: true`. Outputs record the actual clipping setting and geometry policy.
These corrections can change results on the resize/z-score fallback.

## Random streams: sha256-record-draw-v1

For base seed S, stable record ID R, and zero-based draw D, serialize
`["sha256-record-draw-v1", S, R, D]` using Python JSON with `ensure_ascii=True`
and separators `(",", ":")`. SHA-256 the UTF-8 bytes, interpret the first eight
bytes as an unsigned big-endian integer, and clear the top bit. That integer seeds
a separate PyTorch generator for the record and draw on the inference device.

Use these generators for initial noise, DDPM/DDIM stochastic steps, and re-noising
known channels. Production AsynDiff inference remains batched. The generic adapter
fallback runs individually seeded records and restores the caller's global RNG
states. No Python process-randomized hash is used. A stable record ID is required.

Evaluation JSON records the base `seed` and `seed_scheme`. Saved first-draw sample
payloads and manifest entries record the derived seed and scheme; entries identify
`draw_index: 0`. Training history records `evaluation_seed` and `seed_scheme`;
standalone prediction/media payloads record their base seed and scheme.

Changing batch grouping, record order, or evaluating a record alone preserves its
random stream. GPU kernels, library versions and device changes can still cause
floating-point differences; bitwise cross-device reproducibility is not promised.
PSNR follows the confirmed `native-flux-gt-image-max-mean-db-v1` convention:
for each native-flux CHW image, use `max(GT)` over all channels/pixels as the peak,
compute `20 log10(max(peak, 1e-12)) - 10 log10(max(MSE, 1e-12))`, then average
PSNR equally over images. Both evaluators share the implementation. Training
weights batch means by their image counts, including a short last batch. JSON
and training history record `psnr_protocol`. This preserves the independent
evaluator's existing convention and changes the previously inconsistent training
PSNR. Existing non-PSNR training aggregation and limited-batch selection are unchanged.

Legacy evaluation used `seed + chunk_first_index * 1000 + draw` and consumed a
batch-wide stream. New sampling therefore requires a new result artifact/version;
it must not overwrite or be presented as the same realization as legacy results.

## Nonfinite values and undefined arc measurements

Empty or nonfinite input/target/prediction tensors fail evaluation with context.
Diffusion checks raw model predictions before clipping so infinity cannot be
converted into an apparently valid bounded prediction. Metric overflow also fails.
Reconstruction/CRPS JSON and diffusion history use `allow_nan=False`.

A finite image without an extracted arc is different from a numerical failure.
Arc JSON retains null for undefined metrics and adds `undefined_reason`:
`no_gt_arc`, `no_pred_arc`, or `no_gt_arc+no_pred_arc`. Aggregates include reason
counts, `valid_pair_count`, `undefined_pair_count`, and existing per-metric
`finite_count`. TSV includes pair counts and writes `null` for undefined AEE.
Training history represents the no-evaluated-samples case with null and explicit
`undefined_metrics` reasons. Training requires nonempty training and validation
splits and a finite checkpoint-selection metric.

Arc extraction retains the paper's separate clean-source and full-observation
procedures, union component selection and conditional finite-value aggregation.
No missed-detection penalty or cross-method common-valid-subset restriction is
part of this evaluation protocol.

## Arc and classifier evaluation

Arc geometry supports only `separate_masks`. The source GT image and mask are
passed explicitly. Angular extent is the **angular covering span**: `2 pi` minus
the largest circular gap between detected arc-pixel angles. It is the smallest
circular interval enclosing all retained components, including gaps between them,
and is measured in radians. It is not summed occupied angular coverage.

Reports include conditional AEE, valid/undefined pair counts,
coverage and failure reasons. Every AEE table must include **valid-pair coverage**
alongside AEE: `valid_pair_rate = valid_pair_count / count`, where `count` includes
all evaluated examples. Display the numerator and denominator as well, for example
`2 / 4 (50%)`. This is the population on which AEE is defined; prediction-only
detection rate (`valid_pred_arc_rate`) is a separate quantity. JSON, TSV and the
console summary expose valid-pair coverage. Undefined AEE remains `null`, with
coverage reported even when no pair is valid. No missed-arc penalty is introduced.

| Method | Conditional median AEE (rad) | Valid-pair coverage |
| --- | ---: | ---: |
| Illustrative method | 0.075 | 2 / 4 (50%) |

These values illustrate the reporting format, not benchmark results. Conditional
AEE and coverage must be interpreted together; the AEE median alone does not
describe examples where either arc was not detected.

Classifier scoring requires nonempty data containing both labels. With an
evaluation batch limit, a deterministic proportional stratified subset (seed
1337, at least one per class) replaces prefix truncation. Metrics record selected
IDs and counts. This subset score estimates performance only on that subset;
full evaluation remains the default. Checkpoint selection always uses finite
AUROC and never substitutes accuracy.

Training flips use one spatial transform per sample for every modality and
native target before padding. This preserves alignment for odd padding margins.
Photometric noise/scaling remains disabled in released configs; enabling it
requires a separate scientific choice about input perturbations and targets.

## Saved predictions and metric correspondence

RMSE and PSNR score the mean of the native-flux ensemble. Saving draw 0 with
`--save-pred-sample-normalized` also saves this scored mean without extra draws.
The CLI uses `--pred-normalized-root` for the mean; direct evaluator calls that
supply only a sample root use its `ensemble_mean` subdirectory. Each example
records `pred_normalized_path` for the mean and `pred_sample_normalized_paths`
for draw 0. Mean and sample payloads declare their `prediction_kind`.

The mean is formed in native flux before normalization for storage. CRPS uses
the full ensemble. The downstream arc evaluator's default `sample` field and
the classifier manifest's sample paths continue to select draw 0. Reconstruction
metrics therefore identify the mean explicitly, while downstream metrics identify
a single draw. Previously saved files are not rewritten.
