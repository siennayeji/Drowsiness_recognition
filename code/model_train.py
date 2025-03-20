import torch
import torchvision.transforms as transforms
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

# ✅ 한글 파일명을 지원하는 이미지 로드 함수
def imread_unicode(img_path):
    try:
        img_array = np.fromfile(img_path, np.uint8)  # 한글 경로 지원
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"🚨 이미지 로드 실패: {img_path}, 오류: {e}")
        return None

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 📌 Custom Dataset
class DrowsinessDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['ImagePath']
        label = int(self.data.iloc[idx]['Label'])

        # ✅ 한글 경로 지원하는 함수로 이미지 로드
        image = imread_unicode(img_path)

        if image is None:
            print(f"🚨 이미지 로드 실패: {img_path}")
            return self.__getitem__((idx + 1) % len(self.data))  # 다른 샘플로 대체

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label

# 📌 DataLoader 설정
train_dataset = DrowsinessDataset("train.csv", transform=transform)
val_dataset = DrowsinessDataset("val.csv", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 📌 CNN + LSTM 모델
class DrowsinessModel(nn.Module):
    def __init__(self):
        super(DrowsinessModel, self).__init__()

        # CNN 기반 특징 추출
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # LSTM (시계열 학습)
        self.lstm = nn.LSTM(input_size=64 * 28 * 28, hidden_size=128, num_layers=1, batch_first=True)

        # 최종 분류기
        self.fc = nn.Linear(128, 3)  # 3개의 클래스 (하품, 졸음, 정상)

    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN 특징 추출
        x = self.cnn(x)
        x = x.view(batch_size, 1, -1)  # LSTM 입력을 위한 형태 변환
        
        # LSTM 학습
        x, _ = self.lstm(x)

        # 최종 분류
        x = self.fc(x[:, -1, :])
        return x

# 📌 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessModel().to(device)

# 손실 함수 & 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📌 학습 루프
num_epochs = 5 # 원하는 에포크 수 설정
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"🔹 Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

print("🎯 학습 완료!")
# 📌 학습된 모델 저장
torch.save(model.state_dict(), "drowsiness_model.pth")
print("💾 모델 저장 완료: drowsiness_model.pth")

# 모델 평가 모드
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 정확도 계산
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Validation Accuracy: {accuracy * 100:.2f}%")