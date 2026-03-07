import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ringaug",
        description="Run RingAug polygon augmentation on LabelMe image/json datasets.",
    )

    parser.add_argument("--img", "--img-dir", dest="img_dir", required=True, help="Directory containing source images.")
    parser.add_argument("--json", "--json-dir", dest="json_dir", required=True, help="Directory containing source LabelMe JSON files.")

    parser.add_argument(
        "--save",
        "--save-dir",
        dest="save_dir",
        default="./aug_result",
        help="Output root directory. Defaults to ./aug_result",
    )
    parser.add_argument(
        "--index-json-dir",
        default="",
        help="Optional indexed JSON output directory. Defaults to <save>/augmented_index_json",
    )

    parser.add_argument("--num-per-image", type=int, default=2, help="Number of augmentations per source image.")

    parser.add_argument("--crop", nargs=2, type=float, metavar=("MIN", "MAX"), default=[0.8, 0.9])
    parser.add_argument("--rotation", nargs=2, type=float, metavar=("MIN", "MAX"), default=[-30.0, 30.0])
    parser.add_argument("--scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=[0.7, 1.3])
    parser.add_argument("--translate", nargs=2, type=float, metavar=("MIN", "MAX"), default=[-0.1, 0.1])
    parser.add_argument("--brightness", nargs=2, type=float, metavar=("MIN", "MAX"), default=[-0.1, 0.1])
    parser.add_argument("--contrast", nargs=2, type=float, metavar=("MIN", "MAX"), default=[-0.1, 0.1])

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
    parser.add_argument("--debug", action="store_true", help="Enable debug logs in the augmentor.")

    return parser


def _validate_range(name: str, values: Sequence[float]) -> tuple[float, float]:
    low = float(values[0])
    high = float(values[1])
    if low > high:
        raise ValueError(f"{name}: minimum cannot be greater than maximum ({low} > {high})")
    return (low, high)


def build_runtime_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    try:
        crop = _validate_range("crop", args.crop)
        rotation = _validate_range("rotation", args.rotation)
        scale = _validate_range("scale", args.scale)
        translate = _validate_range("translate", args.translate)
        brightness = _validate_range("brightness", args.brightness)
        contrast = _validate_range("contrast", args.contrast)
    except ValueError as err:
        parser.error(str(err))

    img_dir = Path(args.img_dir)
    json_dir = Path(args.json_dir)
    save_root = Path(args.save_dir)

    save_img_dir = save_root / "images"
    save_json_dir = save_root / "json"
    save_index_json_dir = Path(args.index_json_dir) if args.index_json_dir else save_root / "augmented_index_json"

    augmentation_params = {
        "crop_scale_range": crop,
        "angle_limit": rotation,
        "p_rotate": args.p_rotate,
        "p_flip_h": args.p_flip_h,
        "p_flip_v": args.p_flip_v,
        "p_affine": args.p_affine,
        "scale_limit": scale,
        "translate_limit": translate,
        "p_crop": args.p_crop,
        "brightness_limit": brightness,
        "contrast_limit": contrast,
        "p_brightness": args.p_brightness,
        "contour_simplify_epsilon": args.contour_simplify_epsilon,
        "min_component_area": args.min_component_area,
        "min_mask_pixel_area": args.min_mask_pixel_area,
        "random_aug_per_image": args.random_aug_per_image,
    }

    return {
        "img_dir": img_dir,
        "json_dir": json_dir,
        "save_root": save_root,
        "save_img_dir": save_img_dir,
        "save_json_dir": save_json_dir,
        "save_index_json_dir": save_index_json_dir,
        "num_per_image": args.num_per_image,
        "debug": args.debug,
        "augmentation_params": augmentation_params,
    }


def print_run_summary(runtime: dict) -> None:
    params = runtime["augmentation_params"]
    print("Image Dir:", runtime["img_dir"])
    print("Json Dir:", runtime["json_dir"])
    print("Chosen Augmentation Parameters:")
    print("crop:", list(params["crop_scale_range"]))
    print("rotation:", list(params["angle_limit"]))
    print("scale:", list(params["scale_limit"]))
    print("translate:", list(params["translate_limit"]))
    print("brightness:", list(params["brightness_limit"]))
    print("contrast:", list(params["contrast_limit"]))
    print("p_rotate:", params["p_rotate"])
    print("p_flip_h:", params["p_flip_h"])
    print("p_flip_v:", params["p_flip_v"])
    print("p_affine:", params["p_affine"])
    print("p_crop:", params["p_crop"])
    print("p_brightness:", params["p_brightness"])
    print("contour_simplify_epsilon:", params["contour_simplify_epsilon"])
    print("min_component_area:", params["min_component_area"])
    print("min_mask_pixel_area:", params["min_mask_pixel_area"])
    print("random_aug_per_image:", params["random_aug_per_image"])
    print("num_per_image:", runtime["num_per_image"])
    print("Save_dir:", runtime["save_root"])
    print("Output Images:", runtime["save_img_dir"])
    print("Output Json:", runtime["save_json_dir"])
    print("Output Indexed Json:", runtime["save_index_json_dir"])
