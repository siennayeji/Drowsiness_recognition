import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import glob
from sklearn.model_selection import train_test_split
import cv2
import dlib
import pyrealsense2 as rs
import numpy as np

if torch.cuda.is_available():
    torch.set_default_device("cuda")

# ✅ GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # ✅ CNN 모델 성능 최적화
torch.cuda.empty_cache()  # ✅ 불필요한 캐시 정리

print(f"💻 Using device: {device}")
torch.backends.cudnn.benchmark = True  # ✅ GPU 성능 최적화
torch.cuda.empty_cache()  # ✅ GPU 캐시 정리

# ✅ Dlib 얼굴 검출기 로드
detector = dlib.get_frontal_face_detector()

# ✅ RealSense 설정 (RGB 스트림만 사용)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB 스트림만 활성화
pipeline.start(config)

def detect_faces():
    """ RealSense 카메라에서 프레임을 받아 얼굴을 검출하고 좌표를 반환하는 함수 """
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        return None, None

    # RGB 이미지 변환
    color_image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = detector(gray)

    return color_image, faces  # (이미지, 얼굴 리스트) 반환

def release_camera():
    """ 카메라 종료 함수 """
    pipeline.stop()

# ✅ 데이터 로드
json_dir = "C:/data/json"  # JSON 파일 경로
image_dir = "C:/data/image"  # 이미지 파일 경로

def load_json_labels(json_dir, image_dir):
    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
    print(f"📂 Found {len(json_files)} JSON files")

    dataset = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        image_name = data.get("FileInfo", {}).get("FileName")  
        relative_path = os.path.relpath(json_file, json_dir)  
        image_path = os.path.join(image_dir, relative_path).replace("\\", "/")  
        image_path = image_path.rsplit(".", 1)[0] + ".jpg"  

        exists = os.path.exists(image_path)  
        if exists:
            dataset.append((json_file, image_path, data))

    print(f"✅ Loaded {len(dataset)} items")
    if len(dataset) == 0:
        raise ValueError("🚨 데이터가 없습니다! JSON 또는 이미지 경로를 확인하세요.")
    
    return dataset

data = load_json_labels(json_dir, image_dir)

# ✅ 데이터셋 분할
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"✅ Train dataset size: {len(train_data)}")
print(f"✅ Validation dataset size: {len(val_data)}")

# ✅ 이미지 변환 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ✅ 데이터셋 정의
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
            return None  

        if self.transform:
            image = self.transform(image)

        return image, label 

def collate_fn(batch):
    batch = [item for item in batch if item is not None]  
    return torch.utils.data.default_collate(batch) if batch else None  

train_dataset = DrowsinessDataset(train_data, transform=transform)
val_dataset = DrowsinessDataset(val_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                          collate_fn=collate_fn, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, 
                        collate_fn=collate_fn, pin_memory=True, num_workers=4)

# ✅ CNN + LSTM 모델 정의
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

# ✅ 모델 학습 함수
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()
            images = images.unsqueeze(1)  

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')
    
    torch.save(model.state_dict(), 'model.pth')
    print("✅ Model training complete and saved!")

model = DrowsinessModel().to(device)
train_model(model, train_loader, val_loader)

print(f"GPU 사용 메모리: {torch.cuda.memory_allocated(device) / 1024 / 1024:.2f} MB")
print(f"GPU 캐시 메모리: {torch.cuda.memory_reserved(device) / 1024 / 1024:.2f} MB")
