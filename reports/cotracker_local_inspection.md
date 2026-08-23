# Local CoTracker3 inspection

Inspection date: 2026-08-23

## Repository state

- Repository: adjacent checkout `../co-tracker`
- Upstream: `facebookresearch/co-tracker`
- Commit: `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`
- Branch/upstream: `main` / `origin/main`
- Tracked-file diff: none on the local workstation
- Preserved untracked paths: `checkpoints/`, `compare_video_strongmotion.py`,
  `demo2.py`, `myplot.py`, and `online_demo2.py`
- Policy: VideoEEW does not edit, clean, stage, or commit the adjacent checkout.

The separate checkout on server 242 is at the same commit but has user-owned
tracked and untracked changes. It is treated only as an execution dependency;
VideoEEW will not modify or synchronize that checkout.

## Checkpoints

| Checkpoint | SHA-256 | Approx. size |
|---|---|---:|
| `baseline_offline.pth` | `da09bbac871f7398e5b29c4de5213652658949737bc158840b101678ba8ad1df` | 98 MiB |
| `baseline_online.pth` | `8b30b2f239de9987323b729d9115cc5163720a07348a97d045095cd9ebdb7b3a` | 97 MiB |
| `efficientnet_b0_rwightman-7f5810bc.pth` | `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934` | 21 MiB |
| `scaled_offline.pth` | `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834` | 98 MiB |
| `scaled_online.pth` | `205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218` | 97 MiB |

The initial online baseline uses `scaled_online.pth`.

## Confirmed model parameters

| Property | Local source value | Source boundary |
|---|---:|---|
| online window length | 16 | `CoTrackerOnlinePredictor` constructor |
| online step | 8 | `model.window_len // 2` |
| model resolution | 384 × 512 | `CoTrackerThreeBase` |
| encoder stride | 4 | `build_cotracker` |
| correlation radius | 3 | `build_cotracker` |
| correlation pyramid levels | 4 | `CoTrackerThreeBase` |
| predictor inference iterations | 6 | hard-coded predictor call |
| latent feature dimension | 128 | `CoTrackerThreeBase` |
| update transformer hidden size | 384 | `EfficientUpdateFormer` construction |

The predictor currently hard-codes six iterations and its visibility decision.
VideoEEW must therefore call the underlying model when a configured iteration
count is required, or explicitly verify any minimal upstream patch.

## Module and operator boundaries

- Feature encoder: `CoTrackerThreeBase.fnet`, a `BasicEncoder`.
- Feature pyramid: encoder output plus three `avg_pool2d` levels.
- Correlation sampling: `get_correlation_feat` calls `bilinear_sampler`, which
  delegates to `torch.nn.functional.grid_sample` for 4-D or 5-D input.
- Correlation contraction: `forward_window` uses
  `torch.einsum("btnhwc,bnijc->btnhwij", ...)`.
- Update module: `CoTrackerThreeBase.updateformer`.
- Visibility and confidence are predicted separately and multiplied by the
  public online predictor before thresholding at 0.6.

At model resolution, the encoder feature maps have the following expected
shapes for batch `B` and chunk length `T`:

```text
level 0: [B, T, 128, 96, 128]
level 1: [B, T, 128, 48,  64]
level 2: [B, T, 128, 24,  32]
level 3: [B, T, 128, 12,  16]
```

## Online state audit

`init_video_online_processing()` creates:

```text
online_ind
online_track_feat[4]
online_track_support[4]
online_coords_predicted
online_vis_predicted
online_conf_predicted
```

For `N` query points, each cached track feature is approximately
`[B, 1, N, 128]`, and each support feature is approximately
`[B, 49, N, 128]`. The coordinate/visibility/confidence prediction tensors
retain historical frames and are padded by eight frames on successive online
calls. This is linear history growth in the public implementation.

VideoEEW will bound it non-invasively by periodic, logged reseeding in its
adapter. A later internal cache rewrite is allowed only with numerical
regression evidence and an upstream patch report.

## Deployment implications

`GridSample` and `Einsum` are present in the exact online call chain. Full-model
RKNN conversion is therefore not assumed. The first export boundary is the
convolutional `fnet` encoder; correlation sampling remains a CPU/GPU fallback
until a verified alternative exists.

