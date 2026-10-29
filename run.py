"""Generate camera-controlled views from a single image.

    python run.py --image assets/examples/image000.png --config configs/orbit_left.yaml
    python run.py --image my_photo.jpg --config configs/*.yaml          # every camera action

Each image and camera action is a separate run, saved to <out>/<image>/<camera>/.
"""

import argparse
import os

from omegaconf import OmegaConf

from camtrolpp.pipeline import CamTrolPP, Settings, save_result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", nargs="+", required=True, help="input image(s)")
    p.add_argument("--config", nargs="+", required=True, help="camera config(s) from configs/")
    p.add_argument("--out", default="outputs")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-epipolar", action="store_true", help="turn off epipolar attention")
    p.add_argument("--no-color-match", action="store_true", help="turn off LAB colour anchoring")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    settings = Settings(seed=args.seed, color_match=not args.no_color_match)
    if args.no_epipolar:
        settings.refine.epipolar_layers = ()

    model = CamTrolPP(args.checkpoints)
    for image in args.image:
        name = os.path.splitext(os.path.basename(image))[0]
        for config in args.config:
            cfg = OmegaConf.load(config)
            out_dir = os.path.join(args.out, name, cfg.camera)
            if args.skip_existing and os.path.exists(os.path.join(out_dir, "meta.json")):
                print(f"[skip] {out_dir}")
                continue
            print(f"[run] {name} {cfg.camera}: {cfg.total} in passes of at most {cfg.per_pass}", flush=True)
            save_result(model.generate(image, cfg.camera, cfg.total, cfg.per_pass, settings), out_dir)
            print(f"[saved] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
