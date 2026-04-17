import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import SimpleFaceCNN

data_path = '/home/zining/GithubProjects/ImageProcessing/dataset'
train_loader, val_loader, test_loader = get_data_loaders(data_dir=data_path, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device: {device}")

model = SimpleFaceCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
best_val_acc = 0.0

print("----Start Training----")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_face_model.pth')

print(f"---Finish Training---\nThe Highest Validate Accuracy Ratio: {best_val_acc:.2f}%")