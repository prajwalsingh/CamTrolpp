"""GPU check that the release code reproduces the paper's orbit results.

Regenerates one MegaScenes orbit from the matched step-size study (E1) with
the full system (four 15 degree passes) and with reprojection only (one
60 degree pass), and compares the 14 scored frames with the frames saved by
the original experiment code.

    python tests/check_equivalence.py --e1 <E1 output root> --out outputs/equivalence
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camtrolpp.pipeline import CamTrolPP, Settings, save_result  # noqa: E402

CONDITIONS = {"full": (True, (23,)), "mech": (False, ())}  # (color matching, epipolar layers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e1", required=True, help="root with full/step_15/<scene> and mech/step_60/<scene>")
    p.add_argument("--e1_code_dir", required=True, help="folder the E1 scene paths are relative to")
    # Verified to reproduce the saved E1 frames bit for bit from a fresh process.
    p.add_argument("--scene", nargs="+", default=["Barcelona_-_Arc_de_Triomf_(2)"], help="scene folder name(s), or all")
    p.add_argument("--cases", nargs="+", default=["full:15", "full:60", "mech:15", "mech:60"], help="condition:degrees per pass")
    p.add_argument("--out", default="outputs/equivalence")
    args = p.parse_args()

    scenes = sorted(os.listdir(os.path.join(args.e1, "full", "step_15"))) if args.scene == ["all"] else args.scene
    model = CamTrolPP()
    ok = True
    for scene, case in [(s, c) for s in scenes for c in args.cases]:
        cond, per_pass = case.split(":")[0], int(case.split(":")[1])
        color, layers = CONDITIONS[cond]
        ref_dir = os.path.join(args.e1, cond, f"step_{per_pass}", scene)
        image = os.path.normpath(os.path.join(args.e1_code_dir, json.load(open(os.path.join(ref_dir, "meta.json")))["scene"]))
        settings = Settings(seed=1, color_match=color)
        settings.refine.epipolar_layers = layers
        result = model.generate(image, "orbit_left", 60.0, per_pass, settings)
        save_result(result, os.path.join(args.out, f"{cond}_{per_pass}", scene))

        new = result["frames"][result["eval_idx"]].astype(np.int16)
        ref = np.stack([np.array(Image.open(os.path.join(ref_dir, "eval_frames", f"{j:02d}.png"))) for j in range(14)]).astype(np.int16)
        pose_ok = np.allclose(result["poses"][result["eval_idx"]], np.load(os.path.join(ref_dir, "camera_Rt_eval.npy")), atol=1e-6)
        diff = np.abs(new - ref)
        same = np.array_equal(new, ref)
        ok &= same and pose_ok
        print(f"[equivalence] {cond} {per_pass} deg/pass, {scene}: poses {'match' if pose_ok else 'DIFFER'}, "
              f"frames {'bit-identical' if same else 'DIFFER'} (max abs diff {diff.max()}, mean {diff.mean():.4f}, "
              f"identical pixels {np.mean(diff == 0) * 100:.2f}%)", flush=True)
    print("EQUIVALENCE_OK" if ok else "EQUIVALENCE_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
