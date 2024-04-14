import os
from glob import glob
from tqdm import tqdm
from skimage import io

def main(base_dir):
    for i in ['train', 'val', 'test']:
        masks_files = glob(os.path.join(base_dir, i, 'masks_patches', '*.tif'))
        for mask_file in tqdm(masks_files):
            mask = io.imread(mask_file)
            image_file = mask_file.replace('mask', 'image')
            if mask.max() == 0:
                os.remove(mask_file)
                os.remove(image_file)
                # print(mask_file)
                # print(image_file)
                # return
    
        
if __name__ == '__main__':
    base_dir = os.path.join('data', 'Fluo-N3DH-SIM+_splitted_filtered')
    base_dir = "/content/drive/MyDrive/Colab Notebooks/3D segmentation/Fluo-N3DH-SIM+_splitted_filtered"
    main(base_dir)