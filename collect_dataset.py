import random
import pandas as pd
from pathlib import Path

seed = 17
random.seed(seed)


def split_dataset(base_dir, val_sz):
    images_paths = base_dir.rglob("images/*.tif")
    images_paths = [str(x.absolute()) for x in images_paths]

    data = {"path": images_paths}
    df = pd.DataFrame(data)
    random.shuffle(images_paths)

    train_idx = len(images_paths) * (1 - val_sz)
    train_idx = int(train_idx)
    test_idx = train_idx + (len(images_paths) - train_idx) // 2
    train_imgs = images_paths[:train_idx]
    valid_imgs = images_paths[train_idx:test_idx]
    test_imgs = images_paths[test_idx:]

    df.loc[df.path.isin(train_imgs), "split"] = "train"
    df.loc[df.path.isin(valid_imgs), "split"] = "valid"
    df.loc[df.path.isin(test_imgs), "split"] = "test"
    df.to_csv(base_dir / "data.csv", index=False)
    print(df.head())
    print("len(df) =", len(df))
    return df


def build_df(base_dir):
    images_paths = base_dir.rglob("*/images_patches/*.tif")
    images_paths = [str(x.absolute()) for x in images_paths]
    data = {"path": images_paths}
    df = pd.DataFrame(data)
    df["split"] = df["path"].apply(lambda x: x.split("/")[-3])
    df.to_csv(base_dir / "data.csv", index=False)
    print(df.head())
    print("len(df) =", len(df))
    return df


if __name__ == "__main__":
    # base_dir = Path("data") / "Fluo-N3DH-SIM+_splitted"
    base_dir = Path("data") / "Fluo-N3DH-SIM+_splitted_filtered"
    build_df(base_dir)
