# Phase A baseline environment

Inspection and baseline date: 2026-08-23 (Asia/Shanghai)

## Local workstation

- OS: Arch Linux, kernel `7.0.3-arch1-2`, x86_64
- CPU: Intel Xeon E5-1620 v3, 4 cores / 8 threads
- RAM: 31 GiB
- Python: 3.11.11 (`general2` environment)
- PyTorch: 2.11.0, CUDA build 13.0
- Torchvision: 0.26.0
- NumPy: 1.26.4
- SciPy: 1.14.1
- OpenCV: 4.12.0.88
- pandas: 2.3.0
- PyYAML: 6.0.2
- psutil: 5.9.0
- GPU status: NVIDIA driver unavailable to the environment
- Initially unavailable: scikit-learn, pytest, ONNX, ONNX Runtime, RKNN

The workstation is used for source development and CPU-only unit/synthetic
tests. Its installed PyTorch CUDA build is not treated as a usable GPU runtime.

## Server 242 baseline environment

- Architecture: x86_64
- GPUs: four Tesla V100S-PCIE-32GB
- Selected GPU: physical index 3, empty at baseline inspection
- Python environment: `/opt/zb/miniconda3/envs/dtbf_zb`
- Python: 3.10.16
- PyTorch: 2.6.0+cu124
- CUDA reported by PyTorch: 12.4
- NumPy: 2.0.1
- imageio: 2.37.0
- `torch.cuda.is_available()`: true
- OpenCV: absent in the shared environment

The alternative `eew` environment has Python 3.12.13 but no PyTorch and is not
used for the tracker baseline. Shared conda environments will not be mutated;
additional packages will be installed into a project-local environment only if
needed.

At inspection time, server CPU load was high and swap was full, while GPU 3 had
4 MiB allocated and zero utilization. Every later benchmark must re-check load
and identify its GPU explicitly.

## Official online predictor repeatability check

- Input: first 16 frames of upstream `assets/apple.mp4`
- Input SHA-256: `c7f48c5cfb1479e1dbc1df2373d5cad4f55c198bbdb379da0ece10087971542a`
- Decoded input shape: `[1, 16, 3, 720, 1296]`
- Checkpoint: `scaled_online.pth`
- Queries: regular 6 × 6 grid, 36 points
- Online window / step: 16 / 8
- Model resolution: 384 × 512
- First CUDA call: 585.601 ms (includes warm-up)
- Second call: 139.482 ms
- Maximum and mean repeated-track difference: 0 px
- Visibility disagreements: 0
- Visible fraction: 0.857639

The warmed call is below the 266.7 ms block-arrival budget for a 30 FPS,
eight-frame step on this V100S. This is a reproducibility baseline, not an
RK3588 result and not yet a full-pipeline latency claim.

