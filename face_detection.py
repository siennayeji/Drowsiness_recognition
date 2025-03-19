import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

# CSV 데이터 로드
csv_file = "C:/dream/facedetection/drowsiness_dataset.csv"

# 데이터셋 클래스
torch.manual_seed(42)  # 재현성을 위한 시드 고정
class DrowsinessDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.image_paths = self.data["ImagePath"].tolist()
        self.labels = self.data["Label"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = int(self.labels[idx])

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"Skipping corrupted image: {img_path}")
            return None

        if self.transform:
            image = self.transform(image)

        return image, label

# 이미지 변환 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 데이터 로드
dataset = DrowsinessDataset(csv_file, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.default_collate(batch) if batch else None

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# CNN + LSTM 모델
torch.manual_seed(42)
class DrowsinessModel(nn.Module):
    def __init__(self):
        super(DrowsinessModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(16 * 64 * 64, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 3)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = self.fc1(x[:, -1, :])
        return x

# 모델 학습 함수
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.unsqueeze(1)  # (batch, seq_len, C, H, W)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("✅ Model training complete and saved!")

# 모델 학습 실행
model = DrowsinessModel()
train_model(model, train_loader, val_loader)
