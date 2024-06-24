import torch
from x_unet import XUnet


def main():
    print("2D")
    unet = XUnet(
        dim = 64,
        channels = 13,
        out_dim = 1,
        dim_mults = (1, 2, 4, 8),
        nested_unet_depths = (7, 4, 2, 1),     # nested unet depths, from unet-squared paper
        consolidate_upsample_fmaps = True,     # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
    )

    img = torch.randn(1, 13, 256, 256)
    out = unet(img) # (1, 3, 256, 256)
    print(img.shape, out.shape)  
    
    print("3D")
    unet = XUnet(
        dim = 64,
        frame_kernel_size = 3,                 # set this to greater than 1
        channels = 13,
        out_dim = 1,
        dim_mults = (1, 2, 4, 8),
        nested_unet_depths = (5, 4, 2, 1),     # nested unet depths, from unet-squared paper
        consolidate_upsample_fmaps = True,     # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
        weight_standardize = True,
    )

    video = torch.randn(1, 13, 10, 128, 128)    # (batch, channels, frames, height, width)
    out = unet(video) # (1, 3, 10, 128, 128)
    print(video.shape, out.shape)  
    
    
if __name__ == "__main__":
    main()
    