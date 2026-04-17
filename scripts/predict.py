import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from model import SimpleFaceCNN
import torch.nn as nn
import csv

test_dir = '../test_data'
output_dir = '../results/test_outputs'
temp_out_dir = '../results'
output_csv = os.path.join(temp_out_dir, 'prediction_results.csv')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = 'best_face_model.pth'

classes = {0: '0none', 1: '1pouting', 2: '2smile', 3: '3openmouth'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device: {device}")

model = SimpleFaceCNN(num_classes=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

model.eval()

transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("---Start testing---")

with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['File Name', 'Prediction', 'Confidence'] )
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()

            img_path = os.path.join(root, file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(face_rgb)

            input_tensor = transforms(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = nn.functional.softmax(outputs[0], dim=0)
                max_prob, predicted_idx = torch.max(probabilities, 0)

                pred_class = classes[predicted_idx.item()]
                confidence = max_prob.item() * 100

            print(f"{file} --> Prediction: [{pred_class}] (Confidence: {confidence:.1f}%)")

            writer.writerow([file, pred_class, f"{confidence:.2f}"])

            result_img = img.copy()
            text = f"{pred_class}: {confidence:.1f}"

            min_dim = min(img.shape[:2])
            font_scale =  max(0.5, min_dim / 300.0)
            thickness = max(1, int(2 * font_scale))

            cv2.putText(result_img, text, (10, int(35 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            save_path = os.path.join(output_dir, f"result_{file}")
            cv2.imwrite(save_path, result_img)

    print("\n---Finish Testing!---")
