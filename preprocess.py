import os
import glob
import h5py
import trimesh
import numpy as np
from typing import Tuple


def ParseDataset(DATA_DIR: str, num_points=2048) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!R]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

if __name__ == '__main__':
    DATA_DIR = "ModelNet10"
    NUM_POINTS = 2048
    train_points,test_points,train_labels,test_labels,class_map = ParseDataset(DATA_DIR,NUM_POINTS)
    with h5py.File("ModelNet10.h5", "w") as f:
        f.create_dataset("train_x", data=train_points)
        f.create_dataset("test_x", data=test_points)
        f.create_dataset("train_y", data=train_labels)
        f.create_dataset("test_y", data=test_labels)
    print("Preprocessing finished.")