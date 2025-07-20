import os
import glob
import numpy as np

class Dataset():
    def __init__(self, person_dir_path):
        self.data_path = "data/prepared"
        self.person_dir = os.path.join(self.data_path, person_dir_path)
        self.images = glob.glob(f"{self.person_dir}\\ct\\slice_*.npy",recursive=True)
        self.masks = glob.glob(f"{self.person_dir}\\mask\\slice_*.npy", recursive=True)
        self.images.sort()
        self.masks.sort()

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = np.load(image_path)        
        mask = np.load(mask_path)

        return image, mask

class LiverDataset(Dataset):
    def __init__(self, person_id):
        super().__init__(f"person_{person_id}\\liver")
   
class TumorDataset(Dataset):
    def __init__(self, person_id):
        super().__init__(f"person_{person_id}\\tumor")

class LiverTumorDataset(Dataset):
    def __init__(self, person_id):
        super().__init__(f"person_{person_id}\\liver_tumor")

class BgOnlyDataset(Dataset):
    def __init__(self, person_id):
        super().__init__(f"person_{person_id}\\background_only")
