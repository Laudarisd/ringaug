from ringaug.helper import build_parser, build_runtime_config, print_run_summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runtime = build_runtime_config(args, parser)

    print_run_summary(runtime)

    from ringaug.augmentor import IndexPreservingPolygonAugmentor

    augmentor = IndexPreservingPolygonAugmentor(debug=runtime["debug"])
    augmentor.augment_dataset(
        data_dir=runtime["img_dir"],
        json_dir=runtime["json_dir"],
        save_img_dir=runtime["save_img_dir"],
        save_json_dir=runtime["save_json_dir"],
        save_index_json_dir=runtime["save_index_json_dir"],
        num_augmentations=runtime["num_per_image"],
        augmentation_params=runtime["augmentation_params"],
    )


if __name__ == "__main__":
    main()
