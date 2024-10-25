# 3D segmentation of nuclei segmentation

# Get the data
1. Download the data
   ```sh
    wget http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-SIM+.zip
    wget http://data.celltrackingchallenge.net/test-datasets/Fluo-N3DH-SIM+.zip
   ```

2. Unzip it
   ```sh
    unzip Fluo-N3DH-SIM+.zip
    # mv Fluo-N3DH-SIM+ path/to/data
   ```

# How to run
1. Clone repo
   ```sh
    git clone https://github.com/asebaq/cell_segmentation.git
   ```
2. Setup requirements
   ```sh
    cd cell_segmentation
    pip install -r requirements.txt
   ```
3. Split the data
   ```sh
    python patchify_data.py
   ```
4. To remove null patches
   ```sh
    cp "data/Fluo-N3DH-SIM+_splitted" "data/Fluo-N3DH-SIM+_splitted_filtered"
    python filter_patches.py
   ```
5. Collect the data in CSV file
   ```sh
    python collect_dataset.py
   ```

## TODOs
### Segmentation Models 3D
- [x] Combine folder `1` abd `2` of Fluo-N3DH-SIM+ data
- [x] Shuffling and splitting the data into train, val and test
- [x] Adjust the inference script for multiple images instead for one image
- [ ] Add the archimedes optimizer to hypertuning the hyperparameter
- [ ] Try processing the ground truth into three classes(borders, cell, and background) multiclass segmentation. (Optional)
- [ ] Use other datasets such as celegans and drosphilla datasets
- [ ] Modify model architecture for more accuracy
- [ ] Add post processing to convert semantic segmentation results into instance segmentation
- [ ] Compare with 3D object instance segmentation models such as cellpose, stardist, panopatic masckrcnn, detectron, and nnunet

### X-Unet
 - [ ] Train in Google Colab
 - [ ] Add block for instance segmentation in the architecture


## New plan
- [ ] Adjust patchify_data.py to work on all dataset: SIM, CE, DRO
- [ ] Modify infer.py to work with ground truth and add more metrics
- [ ] Modify dataset class to work with mix-vision backbone
- [ ] Run model on three dataset
- [ ] Add more loss functions to enhance model performance
- [ ] Setup the Rescale environment
- [ ] Update model to catch up with pytorch lightning

# Comparison of models
We will select some state of the art models and run them with the best hyperparameter on our dataset then compare them with our model
