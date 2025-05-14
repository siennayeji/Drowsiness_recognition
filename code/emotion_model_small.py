# -*- coding: utf-8 -*-

import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from collections import Counter

# ✅ 감정 레이블 매핑
EMOTION_LABELS = {
    "기쁨": 0,
    "분노": 1,
    "놀람": 2,
    "슬픔": 3,
    "중립": 4
}

# ✅ EmotionDataset 클래스
class EmotionDataset(Dataset):
    def __init__(self, image_root, json_root, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.samples = []

        json_files = glob.glob(os.path.join(json_root, "**/*.json"), recursive=True)
        print(f"🔍 감정 데이터 JSON 파일 수: {len(json_files)}")

        for json_path in json_files:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data.get('scene', {}).get('data', []):
                    img_name = item.get('img_name')
                    if not img_name:
                        continue
                    for occupant in item.get('occupant', []):
                        if 'emotion' in occupant and occupant['emotion'] in EMOTION_LABELS:
                            dir1 = img_name[:10]
                            dir2 = img_name[:15]
                            img_path = os.path.join(self.image_root, dir1, dir2, "img", img_name)
                            self.samples.append((img_path, EMOTION_LABELS[occupant['emotion']]))

        print(f"✅ 총 샘플 수 (경로만 저장): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, OSError, FileNotFoundError):
            return None

        if self.transform:
            image = self.transform(image)

        return image, label

# ✅ None 제거용 collate 함수
def skip_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)

# ✅ 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# ✅ 데이터 경로
IMAGE_ROOT = "/home/sienna/sienna_data/image"
JSON_ROOT = "/home/sienna/sienna_data/json"

# ✅ 데이터셋 준비
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = EmotionDataset(IMAGE_ROOT, JSON_ROOT, transform=transform)
print(f"📊 전체 감정 데이터 샘플 수: {len(dataset)}")

# ✅ 빠르고 안전한 감정 라벨 카운트 (이미지 로딩 없이)
label_counter = Counter(label for _, label in dataset.samples)

# 숫자 → 감정 이름 매핑
idx2label = {v: k for k, v in EMOTION_LABELS.items()}
print("\n📊 감정 클래스별 샘플 수:")
for label_idx, count in label_counter.items():
    print(f"{idx2label[label_idx]}: {count}")

# ✅ 샘플 일부만 사용 (최대 5만개)
if len(dataset) > 50000:
    subset_indices = list(range(50000))
    dataset = Subset(dataset, subset_indices)
    print(f"🎯 5만개로 데이터셋 축소: {len(dataset)}개")

# ✅ 데이터셋 분할
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=skip_none_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=skip_none_collate)

print(f"✅ Train DataLoader 길이: {len(train_loader)}")

# ✅ 모델 준비
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)  # 감정 5개 분류

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ 학습 루프
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        if images is None:
            continue
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
