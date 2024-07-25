from tqdm import tqdm
from skimage import io
from pathlib import Path


def main(base_dir):
    for i in ["train", "val", "test"]:
        masks_files = (base_dir / i / "masks_patches").glob("*.tif")
        masks_files = list(masks_files)
        for mask_file in tqdm(masks_files):
            mask = io.imread(mask_file)
            image_file = Path(str(mask_file).replace("mask", "image"))
            if mask.max() == 0:
                mask_file.unlink()
                image_file.unlink()
                # print(mask_file)
                # print(image_file)
                # return


if __name__ == "__main__":
    base_dir = Path("data") / "Fluo-N3DH-SIM+_splitted_filtered"
    # base_dir = "/content/drive/MyDrive/Colab Notebooks/3D segmentation/Fluo-N3DH-SIM+_splitted_filtered"
    main(base_dir)
