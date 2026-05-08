import torch
from data_loader import get_data_loaders
from resnet_model import CustomResNet
import os

data_path = '../dataset' 
model_path = 'best_face_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device: {device}")

_, _, test_loader = get_data_loaders(data_dir=data_path, batch_size=32)

model = CustomResNet(num_classes=4).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Load the best path: {model_path}")
else:
    print(f"Unabel to find {model_path}, please make sure your path is correct")
    exit()

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total

print("-" * 30)
print(f"Total testcases: {total}")
print(f"Test Accuracy: {test_acc:.2f}%")
print("-" * 30)