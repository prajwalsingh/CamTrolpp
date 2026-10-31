# CamTrol++

Code for **Stabilizing Camera-Controlled Novel View Synthesis at Inference Time**.

[[Paper]](https://arxiv.org/abs/2609.03639) [[Project Page]](https://prajwalsingh.github.io/CamTrolpp/)

CamTrol++ turns a single image into a video that follows a chosen camera path,
using a frozen Stable Video Diffusion model and no training. The camera path is
split into short passes, each covering at most 15 degrees of rotation. For every
pass the latest frame is lifted to 3D with monocular depth, warped to the poses
of the pass, inpainted where the warp leaves holes, and refined by SVD with
epipolar attention. The last frame of a pass, color matched to the input,
starts the next pass.

## Setup

```bash
conda create -n camtrolpp python=3.10 -y
conda activate camtrolpp
pip install -r requirements.txt
bash scripts/download_weights.sh        # SVD and SD-1.5 inpainting, about 14 GB
```

ZoeDepth (with its MiDaS backbone) and the OpenCLIP image encoder of SVD are
downloaded automatically on the first run. One GPU with 48 GB of memory was
used for all results. Run every command from the repository root.

## Generate

Each camera action has its own config in `configs/` and is saved as a separate run.

```bash
# the main setting of the paper, a 60 degree orbit in four 15 degree passes
python run.py --image assets/examples/image000.png --config configs/orbit_left.yaml

# every camera action, one run each
bash scripts/run_all_cameras.sh assets/examples/image000.png
```

| Config | Camera motion | Default size |
|---|---|---|
| `orbit_left`, `orbit_right` | turn around the scene while sliding sideways | 60°, 15° per pass |
| `pan_left`, `pan_right` | turn left or right on the spot | 45°, 15° per pass |
| `tilt_up`, `tilt_down` | turn up or down on the spot | 30°, 15° per pass |
| `roll_left`, `roll_right` | rotate around the viewing axis | 30°, 15° per pass |
| `zoom_in`, `zoom_out` | move forward or backward | 0.5 of the centre depth, 0.25 per pass |
| `truck_left`, `truck_right` | slide sideways | 0.5 of the centre depth, 0.25 per pass |
| `pedestal_up`, `pedestal_down` | move up or down | 0.3 of the centre depth, 0.15 per pass |

`total` sets the size of the move and `per_pass` the most motion one diffusion
pass handles. The number of passes is `ceil(total / per_pass)`. A new camera
action only needs a pose function in `camtrolpp/cameras.py` and a config file.

Useful options: `--seed`, `--no-epipolar`, `--no-color-match`, `--skip-existing`.

Each run writes `outputs/<image>/<camera>/` with

- `frames/` all generated frames (13 per pass plus the input frame)
- `video.mp4`
- `poses_w2c.npy` world-to-camera poses (OpenCV convention) and `K.npy` intrinsics at 960x704
- `meta.json` settings, timing and `eval_idx`, the 14 evenly spaced frames used for evaluation

## Evaluate

```bash
python evaluate.py outputs
```

This scores every run with the metrics of the paper, computed at 256x192 on the
14 frames in `eval_idx`. LPIPS-next and CLIPSIM-next measure the change between
consecutive frames, LPIPS-first and CLIPSIM-first the drift from the input. EAE
and Sampson distance measure agreement with the commanded camera (they need
camera translation, so pan, tilt and roll report NaN). FPA measures
motion smoothness and ZMR the fraction of pixels that do not move. A
`metrics.json` is written next to every run.

The code reproduces the paper's numbers to within rounding. A fresh run of all
20 orbit scenes at 15 and 60 degrees per pass gives frames identical to the
paper's results in 39 of 40 cases, and identical metrics at 60 degrees. At 15
degrees one scene differs slightly, which moves the mean EAE from 2.86 to 2.87
and changes the other metrics by less than 0.002.

## Tests

```bash
python -m pytest tests/                       # CPU, about 30 s
python tests/check_outputs.py outputs/image000  # sanity checks of generated runs
```

`tests/check_equivalence.py` regenerates an orbit from the paper's step-size
study and checks that the frames are identical to the original results, and
`scripts/slurm/smoke_test.sbatch` runs all of the above on a SLURM cluster.

## Layout

```
camtrolpp/
  cameras.py    camera actions and pass splitting
  warp.py       depth, registration-free warping, inpainting
  color.py      LAB color anchoring
  refine.py     SVD refinement of one pass
  pipeline.py   the full multi-pass generation
  metrics.py    evaluation metrics
configs/        one config per camera action
sgm/            Stable Video Diffusion code with epipolar attention
third_party/    ZoeDepth
```

## Acknowledgements and licenses

The code is released under the MIT license. `sgm/` is adapted from
[Stability AI generative-models](https://github.com/Stability-AI/generative-models)
(MIT) with epipolar attention added in `sgm/modules/attention.py`.
`third_party/ZoeDepth` is [ZoeDepth](https://github.com/isl-org/ZoeDepth) (MIT).
The model weights are not part of this repository and keep their own licenses:
[Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
and [Stable Diffusion 1.5 inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting).
The orbit path follows [WAVE](https://github.com/jwoo-park0/WAVE) and the
warp-and-refine idea builds on CamTrol (Hou et al., 2024).

## Citation

```bibtex
@misc{singh2026stabilizingcameracontrollednovelview,
      title={Stabilizing Camera-Controlled Novel View Synthesis at Inference Time},
      author={Prajwal Singh and Arjun Badola and Seema Kumari and Hajime Nagahara and Shanmuganathan Raman},
      year={2026},
      eprint={2609.03639},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.03639},
}
```
