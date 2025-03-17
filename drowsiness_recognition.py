import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import glob
from sklearn.model_selection import train_test_split

# 데이터 로드
json_dir = "C:/data/json"  # JSON 파일 경로
image_dir = "C:/data/image"  # 이미지 파일 경로

def load_json_labels(json_dir, image_dir):
    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
    print(f"📂 Found {len(json_files)} JSON files")

    dataset = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        image_name = data.get("FileInfo", {}).get("FileName")  # JSON에 저장된 이미지 파일명
        relative_path = os.path.relpath(json_file, json_dir)  # JSON 경로 상대 경로 변환
        image_path = os.path.join(image_dir, relative_path).replace("\\", "/")  # json -> image 변경
        image_path = image_path.rsplit(".", 1)[0] + ".jpg"  # 확장자 변경
        
        exists = os.path.exists(image_path)  # 이미지 존재 여부 확인

        if exists:
            dataset.append((json_file, image_path, data))

    print(f"✅ Loaded {len(dataset)} items")
    if len(dataset) == 0:
        raise ValueError("🚨 모든 데이터가 손실되었습니다! JSON 또는 이미지 경로를 확인하세요.")
    
    return dataset

data = load_json_labels(json_dir, image_dir)

# ✅ 데이터셋이 비어 있는지 확인
if len(data) == 0:
    raise ValueError("🚨 데이터셋이 비어 있습니다! JSON 또는 이미지 경로를 확인하세요.")

print(f"📌 Loaded {len(data)} items")

# 데이터 분할
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# ✅ 훈련 / 검증 데이터 개수 확인
print(f"✅ Train dataset size: {len(train_data)}")
print(f"✅ Validation dataset size: {len(val_data)}")

if len(train_data) == 0:
    raise ValueError("🚨 훈련 데이터셋이 비어 있습니다! 데이터를 확인하세요.")

if len(val_data) == 0:
    raise ValueError("🚨 검증 데이터셋이 비어 있습니다! 데이터를 확인하세요.")

# 이미지 변환 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 데이터셋 정의
class DrowsinessDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = [(img_path, metadata_dict['Annotation']) for json_path, img_path, metadata_dict in data if os.path.exists(img_path)]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    
print(f"🔍 데이터 구조 확인: {data[:5]}")

# DataLoader 설정
def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # None 값 제거
    return torch.utils.data.default_collate(batch) if batch else None  # 빈 배치 방지

train_dataset = DrowsinessDataset(train_data, transform=transform)
val_dataset = DrowsinessDataset(val_data, transform=transform)

# ✅ train_dataset이 비어 있는 경우 에러 방지
if len(train_dataset) == 0:
    raise ValueError("🚨 train_dataset이 비어 있습니다! JSON과 이미지 파일을 확인하세요.")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# 모델 정의
class DrowsinessModel(nn.Module):
    def __init__(self):
        super(DrowsinessModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 64 * 64, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# 모델 학습 함수
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, torch.tensor(labels, dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'📢 Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
    
    torch.save(model.state_dict(), 'model.pth')
    print("✅ Model training complete and saved.")

# 모델 학습 수행
model = DrowsinessModel()
train_model(model, train_loader, val_loader)
