import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import glob
from sklearn.model_selection import train_test_split
import face_detection
import cv2

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

# 데이터 분할
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# ✅ 훈련 / 검증 데이터 개수 확인
print(f"✅ Train dataset size: {len(train_data)}")
print(f"✅ Validation dataset size: {len(val_data)}")

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
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, IOError):
            print(f"🚨 Skipping corrupted image: {img_path}")
            return None  # 손상된 이미지는 건너뜀

        if self.transform:
            image = self.transform(image)

        return image, label 

# DataLoader 설정
def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # None 값 제거
    return torch.utils.data.default_collate(batch) if batch else None  # 빈 배치 방지

train_dataset = DrowsinessDataset(train_data, transform=transform)
val_dataset = DrowsinessDataset(val_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)


# CNN + LSTM 모델 정의
class DrowsinessModel(nn.Module):
    def __init__(self):
        super(DrowsinessModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(16 * 64 * 64, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 2)
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
            images = images.unsqueeze(1)  # LSTM 입력을 위해 (batch, seq_len, C, H, W) 형태로 변환
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')
    torch.save(model.state_dict(), 'model.pth')
    print("✅ Model training complete and saved!")

# 모델 학습 수행
model = DrowsinessModel()
train_model(model, train_loader, val_loader)

# 실시간 얼굴 인식 및 졸음 감지
recent_frames = []

while True:
    color_image, faces = face_detection.detect_faces()  # ✅ 얼굴 검출 함수 사용

    if color_image is None:
        continue

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = color_image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = torch.tensor(face_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        recent_frames.append(face_img)

        if len(recent_frames) > 10:
            recent_frames.pop(0)

        if len(recent_frames) == 10:
            input_tensor = torch.stack(recent_frames).unsqueeze(0)
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = "Drowsy" if predicted.item() == 1 else "Awake"
            color = (0, 0, 255) if predicted.item() == 1 else (0, 255, 0)
            cv2.putText(color_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Drowsiness Detection", color_image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

face_detection.release_camera()  # ✅ 카메라 종료
cv2.destroyAllWindows()
