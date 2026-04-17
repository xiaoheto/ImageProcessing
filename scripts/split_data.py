import os
from pathlib import Path
import random
import shutil

random.seed(42)

base_dir = Path("/home/zining/GithubProjects/ImageProcessing")
init_data = base_dir / "init_data"
dataset_dir = base_dir / "dataset"

train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1

classes = [d for d in os.listdir(init_data) if os.path.isdir(init_data / d)]

for split in ['train', 'test', 'val']:
    for cls in classes:
        (dataset_dir / split / cls).mkdir(parents=True, exist_ok = True)

for cls in classes:
    cls_path = init_data / cls
    images = [f for f in os.listdir(cls_path) if os.path.isfile(cls_path / f)]
    random.shuffle(images)

    total_imgs = len(images)
    train_end = int(total_imgs * train_ratio)
    test_end = train_end + int(total_imgs * test_ratio)

    train_images = images[:train_end]
    test_images = images[train_end:test_end]
    val_images = images[test_end:]

    def copy_imgs(img_list, split_name):
        for img in img_list:
            src = cls_path / img
            dst = dataset_dir / split_name / cls / img
            shutil.copy(src, dst)

    copy_imgs(train_images, 'train')
    copy_imgs(test_images, 'test')
    copy_imgs(train_images, 'val')

    print(f"Finish spliting {cls} type.\nTotal image count: {total_imgs} \nTrain image count: {len(train_images)} \nTest image count: {len(test_images)} \nValidate image count: {len(val_images)}")

print("Finish spliting")
