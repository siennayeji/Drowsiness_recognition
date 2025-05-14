import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ✅ 감정 라벨 매핑
EMOTION_LABELS = {
    "기쁨": 0, "당황": 1, "분노": 2, "불안": 3,
    "상처": 4, "슬픔": 5, "중립": 6
}

# ✅ 경로
IMAGE_DIR = "/home/sienna/sienna_data/092.한국인_감정인식을_위한_복합_영상_데이터/01.데이터/1.Training/원천데이터"
JSON_DIR = "/home/sienna/sienna_data/092.한국인_감정인식을_위한_복합_영상_데이터/01.데이터/1.Training/라벨링데이터"

# ✅ 전처리 (RGB 기준)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ✅ 커스텀 Dataset
class EmotionDataset(Dataset):
    def __init__(self, image_root, json_root, transform=None):
        self.data = []
        self.transform = transform
        self.class_counts = {emotion: 0 for emotion in EMOTION_LABELS}

        for emotion in EMOTION_LABELS:
            json_path = os.path.join(json_root, f"{emotion}_unzipped", f"img_emotion_training_data({emotion}).json")
            image_dir = os.path.join(image_root, f"{emotion}_unzipped")

            if not os.path.exists(json_path) or not os.path.exists(image_dir):
                print(f"❌ {emotion} 데이터 경로 누락 - 건너뜀")
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            for item in records:
                filename = item["filename"]
                if "/" in filename:
                    filename = os.path.basename(filename)
                box = item["annot_A"]["boxes"]
                path = os.path.join(image_dir, filename)

                if os.path.exists(path):
                    self.data.append((path, box, EMOTION_LABELS[emotion]))
                    self.class_counts[emotion] += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, box, label = self.data[idx]
        try:
            image = Image.open(path).convert("RGB")
            cropped = image.crop((box["minX"], box["minY"], box["maxX"], box["maxY"]))
            if self.transform:
                cropped = self.transform(cropped)
            return cropped, label
        except Exception as e:
            print(f"❌ 이미지 로딩 실패: {path} | 에러: {e}")
            return None

# ✅ 데이터 로딩
full_dataset = EmotionDataset(IMAGE_DIR, JSON_DIR, transform=transform)

# ✅ 감정별 수 출력
print("\n📊 감정별 유효 이미지 수:")
for emotion, count in full_dataset.class_counts.items():
    print(f"{emotion}: {count}개")
print(f"\n✅ 학습에 사용되는 총 데이터 수: {len(full_dataset)}")

# ✅ 데이터 분할
train_len = int(len(full_dataset) * 0.8)
val_len = len(full_dataset) - train_len
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate)

# ✅ 모델 정의 (RGB 채널 대응)
class DeepCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ✅ 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ 학습 루프
EPOCHS = 30
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "korean_emotion_model_rgb.pth")
        print("✅ 모델 저장 완료: korean_emotion_model_rgb.pth")

print(f"\n🎯 최고 검증 정확도: {best_acc:.2f}%")
