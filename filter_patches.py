import os
from glob import glob
from tqdm import tqdm
from skimage import io

def main(base_dir):
    for i in ['train', 'val', 'test']:
        masks_files = glob(os.path.join(base_dir, i, 'masks', '*.tif'))
        for mask in tqdm(masks_files):
            mask = io.imread(mask)
            if mask.max() == 0:
                os.remove(mask)
                image = os.path.join(base_dir, i, 'images', os.path.basename(mask))
                os.remove(image)
    
        
if __name__ == '__main__':
    base_dir = os.path.join('data', 'Fluo-N3DH-SIM+_splitted_filtered')
    base_dir = os.path.join('content', 'My Drive', '3D Segmentation', 'Fluo-N3DH-SIM+_splitted_filtered')
    main(base_dir)