import albumentations as A
import numpy as np

class ScaleHU(A.ImageOnlyTransform):
    def __init__(self, min_hu=-100, max_hu=400, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.min_hu = min_hu
        self.max_hu = max_hu

    def apply(self, image, **params):
        image = np.clip(image, self.min_hu, self.max_hu)
        image = (image - self.min_hu) / (self.max_hu - self.min_hu)
        return image
