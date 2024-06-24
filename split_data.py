from pathlib import Path


def main():
    # test_set_images = Path("test_set_simulated_nuclei.txt")
    test_set_images = Path("test_set_stem_cells_gow1.txt")
    with open(test_set_images, "r") as f:
        line = f.readline()
        line = line.split(",")

    # base_dir = Path("data_2d_split")
    base_dir = Path("data_2d_gow1_split")
    for folder in ["images", "labels"]:
        for split in ["train", "test"]:
            (base_dir / folder / split).mkdir(exist_ok=True)

    images_path = (base_dir / "images").glob("*.tif")
    for image_path in images_path:
        image_name = image_path.stem
        label_name = str(image_path.name).replace("img", "label")
        if image_name in line:
            image_path.rename(base_dir / "images" / "test" / image_path.name)
            (base_dir / "labels" / label_name).rename(
                base_dir / "labels" / "test" / label_name
            )
        else:
            image_path.rename(base_dir / "images" / "train" / image_path.name)
            (base_dir / "labels" / label_name).rename(
                base_dir / "labels" / "train" / label_name
            )


if __name__ == "__main__":
    main()
