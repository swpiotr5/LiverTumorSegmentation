import albumentations as A
from ScaleHU import ScaleHU

def get_train_transform():
    return A.Compose([
        ScaleHU(min_hu=-100, max_hu=400),
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, 
                           rotate_limit=15, p=0.5),
    ])
