# Data Schema

The release expects normalized Zarr shards and JSON manifests. Dataset creation and simulation code are not part of this repository.

## Modalities

- `lsst`: 5 channels, bands `g,r,i,z,y`
- `euclid`: 1 channel, band `VIS`
- `roman`: 3 channels, bands `F106,F129,F158`

Training and inference loaders resize tensors to the model grid declared in each config, usually `64 x 64`, and keep native-resolution targets for metric computation.

## Nonlens Dataset

Expected root:

```text
data/regular/
  normalization_metadata.json
  run_000/shards/manifest_lre_nonlens.json
  run_000/shards/nonlens/nonlens_000000.zarr
```

Each nonlens shard stores:

- `id`: sample identifier
- `index`: global sample index
- `label`: `0`
- `images/{lsst,euclid,roman}`: normalized observations
- optional scene decomposition groups used only by analysis code

## Lens Dataset

Expected root:

```text
data/lens/
  normalization_metadata.json
  run_000/shards/manifest_lens.json
  run_000/shards/lens/lens_000000.zarr
```

Each lens shard stores:

- `id`: sample identifier
- `index`: global sample index
- `label`: `1`
- `images/{lsst,euclid,roman}`: normalized observations
- `source_lensed_clean_psf/roman`: Roman lensed-source target used by angular extent error
- `deflector_clean_psf/roman`: Roman deflector image used to estimate the lens center

## Manifest Records

The loaders expect each manifest to contain a `results` list. Records with `status: ok` are indexed. Each record must include `path`, pointing to a shard path readable from the current machine. Relative paths are preferred for portable dataset releases.

## Normalization

`normalization_metadata.json` stores per-band signed capped asinh parameters and is read by all released loaders. The same normalization metadata should be used for common/nonlens training, lens evaluation, imputation metrics, and downstream Roman classification.


### Shard path resolution and split caches

Relative manifest shard paths resolve under the explicit data root, never the
current working directory. Legacy prefixes before `run_*` are stripped only as
a fallback when the root-relative candidate is absent. Resolved paths are stored
as absolute paths; intentional external absolute paths remain supported for trusted
manifests. Sample index schema 2 stores requested counts/runs/manifest name in
`build_settings` separately from resolved counts in `summary`. Old caches rebuild
once; an unrequested remainder assigned to test does not cause repeated rebuilds.

Training additive Gaussian noise is configured with `augment_noise_prob` (default
0). The legacy `augment_shift_prob` key remains an alias when the new key is
absent; if both are present, `augment_noise_prob` takes precedence.
