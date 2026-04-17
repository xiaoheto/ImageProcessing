import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=val_test_transforms)

    print(f"Discover {len(train_dataset)} images for training")

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
