import os
import glob
import numpy as np

class Dataset():
    def __init__(self, base_dir="../data/prepared", transformation=None, data_type = ['liver', 'liver_tumor', 'tumor'], limit_patients=None):
        self.images = []
        self.masks = []
        self.transformation = transformation
        self.data_type = data_type

        person_dirs = [d for d in sorted(os.listdir(base_dir)) if "person" in d]
        if limit_patients is not None:
            person_dirs = person_dirs[:limit_patients]

        for person_dir in person_dirs:
            for data_type in self.data_type:
                data_type_dir = os.path.join(base_dir, person_dir, data_type)
                is_ct_and_mask = os.listdir(data_type_dir)
                if "ct" in is_ct_and_mask:
                    ct_dir = os.path.join(data_type_dir, "ct")
                    for ct_scan in sorted(os.listdir(ct_dir)):
                        ct_path = os.path.join(ct_dir, ct_scan)
                        try:
                            arr = np.load(ct_path)
                            self.images.append(ct_path)
                        except Exception as e:
                            print(f"Pominięto uszkodzony plik CT: {ct_path}, błąd: {e}")
                if "mask" in is_ct_and_mask:
                    mask_dir = os.path.join(data_type_dir, "mask")
                    for mask in sorted(os.listdir(mask_dir)):
                        mask_path = os.path.join(mask_dir, mask)
                        try:
                            arr = np.load(mask_path)
                            self.masks.append(mask_path)
                        except Exception as e:
                            print(f"Pominięto uszkodzony plik maski: {mask_path}, błąd: {e}")

    def __getitem__(self, idx):
        image = np.load(self.images[idx])
        mask  = np.load(self.masks[idx])

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        if self.transformation:
            augmented = self.transformation(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask

    def __len__(self):
        return len(self.images)

