import os
from tqdm import tqdm
from glob import glob
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
seed = 17
random.seed(seed)
import warnings
warnings.filterwarnings("ignore")
from shutil import copyfile


def merge_image_patches(image_patches, volume_shape, patch_size):
    """
    Merge patches back into the original image.

    Parameters:
        image_patches (list): List of image patches.
        volume_shape (tuple): Tuple specifying the shape of the original volume (depth, height, width).
        patch_size (tuple): Tuple containing the size of the patches (depth, height, width).

    Returns:
        numpy.ndarray: Reconstructed 3D image volume.
    """
    reconstructed_volume = np.zeros(volume_shape, dtype=image_patches.dtype)
    dz, dh, dw = patch_size
    depth, height, width = volume_shape

    idx = 0
    for z in range(0, depth, dz):
        for y in range(0, height, dh):
            for x in range(0, width, dw):
                # Calculate the end indices of the patch
                z_end = min(z + dz, depth)
                if z_end == depth:
                    z = max(depth - dz, 0)
                y_end = min(y + dh, height)
                if y_end == height:
                    y = max(height - dh, 0)
                x_end = min(x + dw, width)
                if x_end == width:
                    x = max(width - dw, 0)
                    
                # Add the patch to the corresponding region in the volume
                reconstructed_volume[z:z_end, y:y_end, x:x_end] = image_patches[idx]
                # plt.imshow(reconstructed_volume[10])
                # plt.show()
                idx += 1

    return reconstructed_volume

def cut_3d_images(image_volume, patch_size):
    """
    Cuts 3D images and masks into patches and pads the remaining patch with zeros.

    Parameters:
        image_volume (numpy.ndarray): 3D numpy array representing the image volume.
        mask_volume (numpy.ndarray): 3D numpy array representing the mask volume.
        patch_size (tuple): Tuple containing the size of the patches (depth, height, width).

    Returns:
        Tuple containing two lists, one for image patches and one for mask patches.
    """

    image_patches = []

    depth, height, width = image_volume.shape

    dz, dh, dw = patch_size

    for z in range(0, depth, dz):
        for y in range(0, height, dh):
            for x in range(0, width, dw):
                # Calculate the end indices of the patch
                z_end = min(z + dz, depth)
                if z_end == depth:
                    z = max(depth - dz, 0)
                y_end = min(y + dh, height)
                if y_end == height:
                    y = max(height - dh, 0)
                x_end = min(x + dw, width)
                if x_end == width:
                    x = max(width - dw, 0)

                # Extract patch
                image_patch = image_volume[z:z_end, y:y_end, x:x_end]
                image_patches.append(image_patch)

    image_patches = np.array(image_patches, dtype=image_volume.dtype)
    return image_patches

def train_test_val(images_paths, masks_paths, base_dir, val_sz=0.1, seed=17):
    image_paths_train, image_paths_test, mask_paths_train, mask_paths_test = train_test_split(
        images_paths, masks_paths, test_size=val_sz, random_state=seed)
    
    image_paths_train, image_paths_val, mask_paths_train, mask_paths_val = train_test_split(
        image_paths_train, mask_paths_train, test_size=0.1, random_state=seed)
    
    print('Train size:', len(image_paths_train))
    print('Val size:', len(image_paths_val))
    print('Test size:', len(image_paths_test))
    assert set(image_paths_train).isdisjoint(set(image_paths_val))
    assert set(image_paths_train).isdisjoint(set(image_paths_test))
    
    base_dir += '_splitted'
    for i in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, i, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, i, 'masks'), exist_ok=True)
    
    for i, j in zip(image_paths_train, mask_paths_train):
        copyfile(i, os.path.join(base_dir, 'train', 'images', os.path.basename(i)))
        copyfile(j, os.path.join(base_dir, 'train', 'masks', os.path.basename(j)))
        
    for i, j in zip(image_paths_val, mask_paths_val):
        copyfile(i, os.path.join(base_dir, 'val', 'images', os.path.basename(i)))
        copyfile(j, os.path.join(base_dir, 'val', 'masks', os.path.basename(j)))
        
    for i, j in zip(image_paths_test, mask_paths_test):
        copyfile(i, os.path.join(base_dir, 'test', 'images', os.path.basename(i)))
        copyfile(j, os.path.join(base_dir, 'test', 'masks', os.path.basename(j)))

def main(base_dir):
    
    images_dir = os.path.join(base_dir, '01')
    masks_dir = os.path.join(base_dir, '01_GT', 'SEG')
    
    images_paths = glob(os.path.join(images_dir, '*.tif'))
    images_paths.sort()
    masks_paths = glob(os.path.join(masks_dir, '*.tif'))
    masks_paths.sort()
    
    train_test_val(images_paths, masks_paths, base_dir)
    
    
    patch_size = (32, 128, 128)
    for split in ['train', 'val', 'test']:
        images_paths = glob(os.path.join(base_dir + '_splitted', split, 'images', '*.tif'))
        images_paths.sort()
        masks_paths = glob(os.path.join(base_dir + '_splitted', split, 'masks', '*.tif'))
        masks_paths.sort()
        
        os.makedirs(os.path.join(base_dir + '_splitted', split, 'images_patches'), exist_ok=True)
        os.makedirs(os.path.join(base_dir + '_splitted', split, 'masks_patches'), exist_ok=True)
        
        for i in tqdm(range(len(images_paths))):
            idx = images_paths[i][-7:-4]
            assert idx == masks_paths[i][-7:-4]
            
            image = io.imread(images_paths[i])
            mask = io.imread(masks_paths[i])
            mask = (mask > 0).astype(mask.dtype)
            
            image_patches = cut_3d_images(image, patch_size)
            # image_rec = merge_image_patches(image_patches, image.shape, patch_size)
            # print(np.allclose(image, image_rec))
            # print(np.array_equal(image, image_rec))
            for j in range(image_patches.shape[0]):
                io.imsave(os.path.join(base_dir + '_splitted', split, 'images_patches', f'image_{idx}_{i}_{j}.tif'), image_patches[j])
                
            mask_patches = cut_3d_images(mask, patch_size)
            for j in range(mask_patches.shape[0]):
                io.imsave(os.path.join(base_dir + '_splitted', split, 'masks_patches', f'mask_{idx}_{i}_{j}.tif'), mask_patches[j])
            
            # print(image_patches.shape)
            # print(mask_patches.shape)
            # return
        
        
    
if __name__ == '__main__':
    base_dir = os.path.join('data', 'Fluo-N3DH-SIM+')
    base_dir = os.path.join('content', 'My Drive', '3D Segmentation', 'Fluo-N3DH-SIM+')
    main(base_dir)
