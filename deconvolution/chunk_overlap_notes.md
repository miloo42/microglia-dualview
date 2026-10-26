# Chunk overlap and chunk size for tiled RL deconvolution

Notes for `dask_tnia_deconv.py` (dask `map_overlap` + tnia `richardson_lucy_cp`, `noncirc=True`).

## How much overlap (`depth`)?

The published rule of thumb is **at least half the PSF size** per side:

- [Dask blog, GPU deconvolution (2020)](https://blog.dask.org/2020/11/12/deconvolution) uses
  `depth = psf.shape // 2` and requires chunks to be at least as big as the PSF. It doesn't
  test whether that overlap leaves artifacts.
- [Domanski et al., MODSIM 2011 (CSIRO)](https://www.mssanz.org.au/modsim2011/A8/domanski.pdf):
  tiled 3D RL on GPUs, guard band "at least half the PSF diameter". *(From a search summary;
  I haven't read the PDF.)*
- [Mourya et al. 2017, arXiv:1705.06603](https://arxiv.org/abs/1705.06603): reportedly an overlap
  "slightly larger than half the size of the local PSF" is enough when the blur doesn't vary
  across the image. *(From a search summary; the method isn't plain RL.)*
- [clij2-fft `richardson_lucy_dask`](https://github.com/clij/clij2-fft) (same author as tnia):
  the default overlap is only 10 px, with the advice to keep it as small as possible while
  avoiding artifacts.
- [Image-Pro / AutoQuant docs](https://help.mediacy.com/Image-Pro/11/en/Content/B_User's%20Guide/Using%20AutoQuant%20Deconvolution/Deconvolution%20Settings.htm):
  no rule based on PSF size; if tiling artifacts appear, "doubling the guardband (and subvolume
  overlap) value often yields more satisfactory results".

**No source found that says a full PSF width is required.** But half a PSF is the reasoning for a
*single* convolution. RL is iterative: each iteration blurs with the PSF and then with its
mirror image, so one iteration already draws on data up to a full PSF width away. Over many
iterations (we use 50) that reach keeps growing, although the far-away contributions shrink
quickly. So:

- half a PSF is the theoretical minimum
- a **full PSF (current `depth = psf.shape[0]` = 75 px) is the safer choice** and is kept
- a smaller overlap should only be used after testing: deconvolve one region as a single
  block and again in chunks, then compare near the seams
- possible saving: set the overlap from where the PSF actually carries its energy (e.g. the
  radius holding 95% of it, as in [arXiv:1505.01684](https://arxiv.org/pdf/1505.01684))
  instead of the full 75 px PSF box

## Chunk size vs GPU memory

With `CHUNK_FACTOR = 3` (core chunk 225 px):

- after the overlap: 225 + 2·75 = 375³
- after tnia's non-circulant padding (+PSF/2 per side, rounded to an FFT-friendly size): ≈ 450³
- RL keeps roughly 8–10 arrays of that size on the GPU → **~6–7 GB**
- only (225/450)³ ≈ **12.5%** of each FFT volume ends up in the output

Memory grows with the FFT volume (≈ N³). On a 16 GB GPU (RTX 2000 Ada), with a target of
~13 GB, the padded edge can be ≈ 450·2^(1/3) ≈ 560, so the **core chunk ≈ 335 px** while keeping
the full-PSF overlap. That gives ≈ 21% useful volume, **~1.7× faster** overall.

Not yet measured; check the real peak with `nvidia-smi` and stay below ~14 GB. On machines
with smaller GPUs, choose the chunk size from `cp.cuda.Device().mem_info`.
