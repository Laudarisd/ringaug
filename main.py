# Run from project root: python main.py --original-img-dir seg-topo-augment/images --original-json-dir seg-topo-augment/json --output-dir seg-topo-augment/augmented --num-per-image 2
import argparse
import importlib.util
from pathlib import Path


def _load_augmentor_module():
    """Dynamically load augmentor module from path because folder name has a hyphen."""
    project_root = Path(__file__).resolve().parent
    candidate_paths = [
        project_root / "seg-topo-augment/polygon_to_aug.py"
    ]
    module_path = next((p for p in candidate_paths if p.exists()), None)
    if module_path is None:
        expected = ", ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(f"Missing augmentor module. Checked: {expected}")

    spec = importlib.util.spec_from_file_location("polygon_to_aug_original", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run polygon topology augmentation from project root.")

    # Input/Output paths.
    parser.add_argument("--original-img-dir", default="seg-topo-augment/images", help="Directory containing source images.")
    parser.add_argument("--original-json-dir", default="seg-topo-augment/json", help="Directory containing source LabelMe JSON files.")
    parser.add_argument("--output-dir", default="seg-topo-augment/augmented", help="Root output directory.")
    parser.add_argument(
        "--index-json-dir",
        default="",
        help="Optional indexed JSON output directory. If empty, uses <output-dir>/augmented_index_json.",
    )

    # How many augmented samples to generate per source image.
    parser.add_argument("--num-per-image", type=int, default=2, help="Number of augmentations per source image.")

    # Augmentation ranges and probabilities.
    parser.add_argument("--crop-scale-min", type=float, default=0.8)
    parser.add_argument("--crop-scale-max", type=float, default=0.9)
    parser.add_argument("--angle-min", type=float, default=-30.0)
    parser.add_argument("--angle-max", type=float, default=30.0)
    parser.add_argument("--scale-min", type=float, default=0.7)
    parser.add_argument("--scale-max", type=float, default=1.3)
    parser.add_argument("--translate-min", type=float, default=-0.1)
    parser.add_argument("--translate-max", type=float, default=0.1)
    parser.add_argument("--brightness-min", type=float, default=-0.1)
    parser.add_argument("--brightness-max", type=float, default=0.1)
    parser.add_argument("--contrast-min", type=float, default=-0.1)
    parser.add_argument("--contrast-max", type=float, default=0.1)

    parser.add_argument("--p-rotate", type=float, default=0.9)
    parser.add_argument("--p-flip-h", type=float, default=0.2)
    parser.add_argument("--p-flip-v", type=float, default=0.1)
    parser.add_argument("--p-affine", type=float, default=0.8)
    parser.add_argument("--p-crop", type=float, default=0.7)
    parser.add_argument("--p-brightness", type=float, default=0.4)

    parser.add_argument("--contour-simplify-epsilon", type=float, default=1.5)
    parser.add_argument("--min-component-area", type=float, default=12.0)
    parser.add_argument("--min-mask-pixel-area", type=float, default=12.0)
    parser.add_argument("--random-aug-per-image", type=int, default=3)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging in augmentor.")

    return parser.parse_args()


def main():
    args = _build_args()

    module = _load_augmentor_module()
    augmentor = module.IndexPreservingPolygonAugmentor(debug=args.debug)

    output_root = Path(args.output_dir)
    save_img_dir = output_root / "images"
    save_json_dir = output_root / "json"
    save_index_json_dir = Path(args.index_json_dir) if args.index_json_dir else output_root / "augmented_index_json"

    # Single dict passed through to augmentor.
    augmentation_params = {
        "crop_scale_range": (args.crop_scale_min, args.crop_scale_max),
        "angle_limit": (args.angle_min, args.angle_max),
        "p_rotate": args.p_rotate,
        "p_flip_h": args.p_flip_h,
        "p_flip_v": args.p_flip_v,
        "p_affine": args.p_affine,
        "scale_limit": (args.scale_min, args.scale_max),
        "translate_limit": (args.translate_min, args.translate_max),
        "p_crop": args.p_crop,
        "brightness_limit": (args.brightness_min, args.brightness_max),
        "contrast_limit": (args.contrast_min, args.contrast_max),
        "p_brightness": args.p_brightness,
        "contour_simplify_epsilon": args.contour_simplify_epsilon,
        "min_component_area": args.min_component_area,
        "min_mask_pixel_area": args.min_mask_pixel_area,
        "random_aug_per_image": args.random_aug_per_image,
    }

    augmentor.augment_dataset(
        data_dir=args.original_img_dir,
        json_dir=args.original_json_dir,
        save_img_dir=save_img_dir,
        save_json_dir=save_json_dir,
        save_index_json_dir=save_index_json_dir,
        num_augmentations=args.num_per_image,
        augmentation_params=augmentation_params,
    )


if __name__ == "__main__":
    main()
