# ✅ CNN + LSTM 학습 코드 (경로 문제 해결 + 데이터 증강 + 검증 정확도 포함)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from data_labeling import SequenceDataset
from glob import glob
from collections import Counter
import json

# 하이퍼파라미터
BATCH_SIZE = 8
EPOCHS = 10
SEQ_LEN = 5
NUM_CLASSES = 2  # 졸음: 1, 정상: 0

# 주요 행동 리스트
action_vocab = [
    "하품", "꾸벅꾸벅졸다", "눈비비기", "눈깜빡이기", "전방주시", "운전하다", "기타"
]
action2idx = {a: i for i, a in enumerate(action_vocab)}
ACTION_DIM = len(action_vocab)

# 라벨 매핑
label_map = {
    "졸음운전": 1,
    "음주운전": 0,
    "물건찾기": 0,
    "통화": 0,
    "휴대폰조작": 0,
    "차량제어": 0,
    "운전자폭행": 0
}

# 전처리 정의
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

aug_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 데이터 경로
json_root_dir = "C:/data/json"
image_root_dir = "C:/data/image"

# 데이터 확인
json_files = glob(os.path.join(json_root_dir, "**", "*.json"), recursive=True)
image_files = glob(os.path.join(image_root_dir, "**", "*.jpg"), recursive=True)
print(f"📁 JSON 파일 개수: {len(json_files)}")
print(f"📁이미지 파일 개수: {len(image_files)}")

# 데이터셋 생성
print("데이터셋 생성 시작")
try:
    full_dataset = SequenceDataset(
        json_root_dir=json_root_dir,
        image_root_dir=image_root_dir,
        label_map=label_map,
        transform=basic_transform,
        action2idx=action2idx,
        use_face=True
    )
    print("데이터셋 생성 완료")
except Exception as e:
    print(f"데이터셋 생성 중 예외 발생: {e}")
    import traceback
    traceback.print_exc()
    exit()

print("데이터셋 길이:", len(full_dataset))
print(f"\n📊 유효한 시퀀스 수 (Dataset 크기): {len(full_dataset)}")
if len(full_dataset) == 0:
    print("❌ 유효한 시퀀스가 0개입니다. JSON과 이미지 경로, 구조를 다시 확인해주세요.")
    exit()

# 라벨 분포 출력
labels = []
for i in range(len(full_dataset)):
    item = full_dataset[i]
    if item is None:
        continue
    _, label, _ = item
    labels.append(label)
print("📊 라벨 분포:", Counter(labels))

# 데이터 분할
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# collate_fn 정의
def skip_broken_collate_fn(batch):
    filtered = [item for item in batch if item is not None]
    if len(filtered) == 0:
        return None, None, None
    seqs, labels, actions = zip(*filtered)
    seqs = torch.stack(seqs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return seqs, labels, actions

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=skip_broken_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=skip_broken_collate_fn)
print(f"Train DataLoader Length: {len(train_loader)}")

# 모델 정의
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, x):
        x = self.cnn(x)
        return x.view(x.size(0), -1)

class DrowsinessCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128 + ACTION_DIM, NUM_CLASSES)

    def forward(self, x, action_vec):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        out = out[:, -1, :]
        out = torch.cat([out, action_vec], dim=1)
        return self.fc(out)

# 학습 루프

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessCNNLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    count = 0

    for seqs, labels, actions in train_loader:
        if seqs is None:
            continue
        seqs, labels = seqs.to(device), labels.to(device)
        action_idx = [action2idx.get(a, action2idx["기타"]) for a in actions]
        action_vec = torch.nn.functional.one_hot(torch.tensor(action_idx), num_classes=ACTION_DIM).float().to(device)

        optimizer.zero_grad()
        outputs = model(seqs, action_vec)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    # 검증 루프
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seqs, labels, actions in val_loader:
            if seqs is None:
                continue
            seqs, labels = seqs.to(device), labels.to(device)
            action_idx = [action2idx.get(a, action2idx["기타"]) for a in actions]
            action_vec = torch.nn.functional.one_hot(torch.tensor(action_idx), num_classes=ACTION_DIM).float().to(device)
            outputs = model(seqs, action_vec)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0
    print(f"✅ Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/count:.4f} | Val Acc: {acc:.4f}")

# 모델 저장
torch.save(model.state_dict(), "cnn_lstm_drowsiness_1.pth")
print("\n💾 모델 저장 완료!")