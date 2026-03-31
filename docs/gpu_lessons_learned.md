# GPU Acceleration on Moissanite: Lessons Learned

Summary of what worked and what didn't getting CuPy GPU acceleration
running on moissanite for the alibz LIBS spectral analysis pipeline.

## Machine Specs

- **moissanite**: 2x Quadro RTX 8000 (48 GB) + 1x RTX 6000 Ada (48 GB)
- **NVIDIA Driver**: 530.30.02 (supports CUDA 12.1)
- **System CUDA toolkit**: 9.1 (ancient, came with Ubuntu 18.04)
- **System Python**: 3.6.9 (too old for alibz, needs >=3.9)
- **Usable Python**: conda `peakyfinder` env with Python 3.11.4

## What Worked

### CuPy 13.4.1 with pip-installed NVRTC

The combination that works:
```bash
pip install cupy-cuda12x==13.4.1
pip install nvidia-cuda-nvrtc-cu12
```

CuPy 13.4.1 bundles CUDA runtime 12.8, which runs fine on the 12.1 driver
(CUDA minor version forward compatibility). But it does NOT bundle NVRTC
(the runtime compiler), which is needed for any JIT-compiled operations
including basic CuPy broadcasting (`a[:, None] - b[None, :]`).

The `nvidia-cuda-nvrtc-cu12` pip package provides `libnvrtc.so.12` at:
```
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/
```

This path must be on `LD_LIBRARY_PATH` at runtime:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
```

Added a conda activation script at
`$CONDA_PREFIX/etc/conda/activate.d/cuda_paths.sh` to do this automatically.

### GPU-accelerated PCA via CuPy SVD

Direct replacement for sklearn PCA:
```python
U, S, Vt = cp.linalg.svd(Xc, full_matrices=False)
```
Works perfectly, gives exact same results as sklearn. This is the biggest
GPU win for the pipeline — SVD on a 25k x 101 matrix is fast on GPU.

### GPU-accelerated batch interpolation

`cp.searchsorted` + linear interpolation for standardizing spectra onto a
common wavelength grid. Works correctly, no accuracy concerns.

### Pseudo-Voigt approximation for GPU Voigt profiles

Thompson pseudo-Voigt using only CuPy array ops (no custom kernels):
- ~0.2% error at peak center
- ~10-15% relative error in the wings (where absolute values are small)
- Good enough for classification, NOT for curve fitting

## What Didn't Work

### CuPy 14.0.1 (latest)

CuPy 14 bundles CUDA runtime 12.9, but the moissanite driver only supports
CUDA 12.1. Result: NVRTC compilation fails with cryptic
`NVRTC_ERROR_INVALID_OPTION` / "invalid value for --gpu-architecture".

**Lesson**: Always check `nvidia-smi` for the driver's max supported CUDA
version. CuPy's bundled runtime version must be <= the driver version.

### CuPy 12.3.0

Doesn't bundle the CUDA runtime at all — expects system CUDA 12.x libs.
Moissanite's system only has CUDA 9.1 toolkit, so CuPy 12.x fails with
`libcudart.so.12: cannot open shared object file`.

**Lesson**: CuPy >=13 bundles the runtime; CuPy <13 requires a matching
system CUDA toolkit installation.

### Custom CUDA Voigt kernel (Humlicek cpf12)

Attempted to write an `ElementwiseKernel` implementing the Faddeeva function
via the Humlicek cpf12 rational approximation. Two problems:

1. **Required NVRTC**: Custom CUDA kernels need the NVRTC compiler, which
   wasn't initially available (see above).

2. **Numerical accuracy**: The Humlicek cpf12 algorithm involves complex
   arithmetic in 4 regions with different rational approximations. Porting
   the complex Horner evaluation to CUDA C introduced subtle sign/coefficient
   errors that were hard to debug (the output was orders of magnitude wrong).
   The algorithm has many coefficients and complex multiply-accumulate steps
   where a single sign flip produces catastrophic errors.

**Lesson**: Don't hand-roll complex special functions in CUDA unless you have
a reference implementation to copy verbatim. The pseudo-Voigt approximation
using standard CuPy ops is much simpler and sufficient for classification.
For exact Voigt evaluation, keep it on CPU with scipy.

### System nvcc (9.1) interfering with CuPy

The system `nvcc` at `/usr/bin/nvcc` is CUDA 9.1, which doesn't know about
modern GPU architectures. CuPy's bundled NVRTC should take precedence, but
having the old nvcc on PATH can cause confusion in error messages.

**Lesson**: CuPy's bundled NVRTC is separate from the system CUDA toolkit.
Don't rely on `nvcc --version` to indicate CuPy compatibility.

### numpy version conflicts

CuPy 13-14 pulls in numpy 2.x, which breaks matplotlib and scipy that were
compiled against numpy 1.x. Required upgrading scipy and matplotlib after
CuPy installation.

**Lesson**: After installing CuPy, always run:
```bash
pip install --upgrade scipy scikit-learn matplotlib
```

## Architecture Decisions

### Where GPU helps most in the pipeline

1. **PCA SVD** — exact, major speedup for large peak matrices
2. **Batch interpolation** — spectrum standardization
3. **Window extraction** — baseline subtraction, normalization
4. **Batch perturbed profile generation** — decomposition analysis

### Where GPU doesn't help

1. **Peak fitting** — `scipy.optimize.least_squares` is CPU-only and
   iterative. Each iteration's Voigt evaluation is too small to benefit
   from GPU transfer overhead.
2. **MILP composition solving** — PuLP solver is CPU-bound.

### Pseudo-Voigt threshold

Set `_GPU_THRESHOLD = 10_000_000` in `voigt.py` to prevent automatic GPU
dispatch during iterative curve fitting (where accuracy matters). The GPU
Voigt is only used when explicitly requested via `use_gpu=True` for large
batch evaluations.

## Reproducible Setup Commands

```bash
# On moissanite:
cd ~/src/alibz
~/anaconda3/envs/peakyfinder/bin/pip install cupy-cuda12x==13.4.1
~/anaconda3/envs/peakyfinder/bin/pip install nvidia-cuda-nvrtc-cu12
~/anaconda3/envs/peakyfinder/bin/pip install --upgrade scipy scikit-learn matplotlib
~/anaconda3/envs/peakyfinder/bin/pip install -e .

# Activation script (one-time):
mkdir -p ~/anaconda3/envs/peakyfinder/etc/conda/activate.d
cat > ~/anaconda3/envs/peakyfinder/etc/conda/activate.d/cuda_paths.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
EOF

# Launch a run:
export LD_LIBRARY_PATH=~/anaconda3/envs/peakyfinder/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib
cd ~/src/alibz
~/anaconda3/envs/peakyfinder/bin/python -u scripts/run_corpus_pca.py \
  /media/mwhittaker/Corpus_One/All_LIBS_Till_20260319 \
  --gpu --out data/corpus_pca_all_libs.pkl
```
