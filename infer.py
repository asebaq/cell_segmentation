import ssl
import torch
import numpy as np
from skimage import io
from models import CellModel
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from patchify_data import cut_3d_images, merge_image_patches

ssl._create_default_https_context = ssl._create_unverified_context


def main(model_name, model_encoder, model_path, base_dir):
    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CellModel(model_name, model_encoder, 1, 1)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.cuda()
    print("Model loaded successfully")
    patch_size = (32, 128, 128)
    images = base_dir.glob("*.tif")
    for img_path in images:
        image = io.imread(img_path)
        # Patchfy
        image_patches = cut_3d_images(image, patch_size)
        image_patches_pred = np.zeros_like(image_patches)
        for i in tqdm(range(len(image_patches))):
            img = image_patches[i]
            img = img.astype(np.float32)
            # Normalize
            img /= img.max()
            img = torch.tensor(img)
            img = img.unsqueeze(0)
            img = img.unsqueeze(0)
            img = img.to(device)

            with torch.no_grad():
                output = model(img)
                output = (output > 0.5).float()
                output = output.squeeze().cpu()
                output = np.uint8(output.numpy())
                image_patches_pred[i] = output
                # plt.imshow(output[25] * 255, cmap="gray")
                # plt.show()

        image_rec = merge_image_patches(image_patches_pred, image.shape, patch_size)
        io.imsave(str(img_path).replace(".tif", "_pred.tif"), image_rec)


if __name__ == "__main__":
    model_name = "Unet"
    model_encoder = "resnet18"
    model_path = (
        "lightning_logs/version_1/checkpoints/Unet-resnet18-epoch080-0.9250.ckpt"
    )
    base_dir = Path("data") / "Fluo-N3DH-SIM+_splitted" / "test" / "images"
    main(model_name, model_encoder, model_path, base_dir)
