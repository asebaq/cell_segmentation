#! /bin/bash
# echo "Downloading Fluo-N3DH-SIM+ dataset"
# wget http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-SIM+.zip
# wget http://data.celltrackingchallenge.net/test-datasets/Fluo-N3DH-SIM+.zip


# echo "Unzipping Fluo-N3DH-SIM+ dataset"
# unzip Fluo-N3DH-SIM+.zip
# # mv Fluo-N3DH-SIM+ path/to/data

# echo Split the data ...
# python patchify_data.py

echo To remove null patches ...
# mkdir -p "data/Fluo-N3DH-SIM+_splitted_filtered"
# cp -r "data/Fluo-N3DH-SIM+_splitted/" "data/Fluo-N3DH-SIM+_splitted_filtered"
python filter_patches.py

echo Collect the data in CSV file ...
python collect_dataset.py