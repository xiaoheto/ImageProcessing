import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32):
    # 1. Training transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # Moved ColorJitter before ToTensor (Standard practice for PIL images)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Validation and Test transforms (Removed redundant eval_transform and fixed semicolon bug)
    val_test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 3. Load datasets
    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=val_test_transforms)

    print(f"Discover {len(train_dataset)} images for training")

    # 4. Create DataLoaders (Replaced hardcoded 32 with the batch_size parameter)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader