# -*- coding: utf-8 -*-
import os, json
from glob import glob
from PIL import Image, UnidentifiedImageError
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms, models

# --- 하이퍼파라미터 ---
BATCH_SIZE   = 8
EPOCHS       = 10
SEQ_LEN      = 5
NUM_CLASSES  = 2
LR           = 1e-4
TEMPERATURE  = 4.0
ALPHA        = 0.7  # KD vs CE 가중치
NUM_WORKERS  = 8    # DataLoader 워커 수

# --- 라벨 매핑 ---
label_map = {
    "졸음운전": 1,
    "음주운전": 0,
    "물건찾기": 0,
    "통화":     0,
    "휴대폰조작": 0,
    "차량제어": 0,
    "운전자폭행": 0
}

# --- 행동 벡터 ---
action_vocab = ["하품","꾸벅꾸벅졸다","눈비비기","눈깜빡이기","전방주시","운전하다","기타"]
action2idx   = {a:i for i,a in enumerate(action_vocab)}
ACTION_DIM   = len(action_vocab)

# --- 전처리 정의 ---
basic_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
aug_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
train_transform = aug_transform
val_transform   = basic_transform

# --- Dataset 정의 (얼굴 크롭, 깨진 이미지는 __getitem__에서 None 리턴) ---
class SequenceDataset(Dataset):
    def __init__(self, json_root, img_root, label_map, transform, use_face=True):
        self.samples   = []
        self.transform = transform
        self.label_map = label_map
        self.use_face  = use_face

        files = glob(os.path.join(json_root, '**', '*.json'), recursive=True)
        for j in files:
            try:
                data = json.load(open(j, 'r', encoding='utf-8'))
                cat  = data.get('scene_info',{}).get('category_name')
                if cat not in label_map: 
                    continue

                frames = data.get('scene',{}).get('data',[])
                if len(frames) != SEQ_LEN:
                    continue

                base  = os.path.relpath(os.path.dirname(j), json_root)\
                            .replace('label','').rstrip('/\\')
                paths, boxes, ok = [], [], True
                for fr in frames:
                    p = os.path.join(img_root, base, 'img', fr['img_name'])
                    if not os.path.exists(p):
                        ok = False
                        break
                    occ = fr.get('occupant',[])
                    if not occ:
                        ok = False
                        break
                    bbox = occ[0].get('face_b_box')
                    if not bbox or len(bbox)!=4:
                        ok = False
                        break
                    paths.append(p)
                    boxes.append(bbox)

                if ok:
                    self.samples.append((paths, boxes, label_map[cat]))
            except Exception:
                # JSON 파싱 오류 등은 무시
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, boxes, label = self.samples[idx]
        seq = []
        for p, box in zip(paths, boxes):
            try:
                x,y,w,h = box
                img = Image.open(p).convert('RGB')\
                         .crop((x, y, x+w, y+h))
                seq.append(self.transform(img))
            except (UnidentifiedImageError, Exception):
                # 깨진 이미지면 이 시퀀스 전체를 무시
                return None
        return torch.stack(seq), label

# ⭐️ 원본 CNNEncoder 클래스를 그대로 import/정의하세요
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

# ✅ TeacherModel 에서 encoder를 CNNEncoder() 로 바꿉니다
class TeacherModel(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = CNNEncoder()    # ← 변경된 부분
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128 + action_dim, NUM_CLASSES)

    def forward(self, x, a):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x).view(B, T, -1)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        return self.fc(torch.cat([last, a], dim=1))

# --- Student 모델 (CNN + LSTM 구조) ---
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Teacher와 동일한 CNN 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        # LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        # 최종 분류기 (action vector 없이 128-dim → 2 classes)
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        # (B*T, C, H, W)로 펼치고 CNN 인코딩
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x).view(B, T, -1)   # (B, T, 64)
        # LSTM
        out, _ = self.lstm(feats)               # (B, T, 128)
        last = out[:, -1, :]                    # 마지막 타임스텝
        return self.fc(last)                    # (B, NUM_CLASSES)

# --- 메인 실행부 ---
if __name__=='__main__':
    # 1) 데이터 준비 & 통계
    json_root = '/home/sienna/sienna_data/json'
    img_root  = '/home/sienna/sienna_data/image'

    full_ds = SequenceDataset(json_root, img_root, label_map, train_transform)
    print(f"✅ 전체 데이터셋 길이: {len(full_ds)}")

    # 깨진 샘플 건너뛸 인덱스
    valid_idxs = []
    labels = []
    for i in range(len(full_ds)):
        item = full_ds[i]
        if item is None:
            continue
        valid_idxs.append(i)
        _, lbl = item
        labels.append(lbl)

    filtered_ds = Subset(full_ds, valid_idxs)
    counter = Counter(labels)
    print(f"📊 라벨 분포: {counter}")

    # 2) Train/Val split
    n_train = int(0.8 * len(filtered_ds))
    n_val   = len(filtered_ds) - n_train
    train_ds, val_ds = random_split(filtered_ds, [n_train, n_val])

    # 3) 불균형 보정용 sampler
    train_labels = [labels[i] for i in train_ds.indices]
    weights = [1.0 / counter[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # 4) DataLoader (I/O 병목 해소)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda b: default_collate([x for x in b if x is not None])
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda b: default_collate([x for x in b if x is not None])
    )

    # 5) Teacher 모델 불러오기
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEACHER_PATH = '/home/sienna/my_ws/code/cnn_lstm_drowsiness_1.pth'
    teacher = TeacherModel(ACTION_DIM).to(device).eval()
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    for p in teacher.parameters():
        p.requires_grad = False

    # 6) Student 학습 준비
    student = StudentModel().to(device)
    ce_loss = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(student.parameters(), lr=LR)

    def evaluate(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = torch.argmax(model(x), dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        return correct / total if total else 0

    # 7) Distillation 학습 루프
    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        student.train()
        sum_loss, cnt = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            a = torch.zeros(x.size(0), ACTION_DIM, device=device)

            with torch.no_grad():
                t_logits = teacher(x, a)
            s_logits = student(x)

            t_soft = nn.functional.softmax(t_logits / TEMPERATURE, dim=1)
            s_logp = nn.functional.log_softmax(s_logits / TEMPERATURE, dim=1)

            loss_kd = kd_loss(s_logp, t_soft) * (TEMPERATURE**2)
            loss_ce = ce_loss(s_logits, y)
            loss    = ALPHA * loss_kd + (1-ALPHA) * loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            cnt      += 1

        val_acc = evaluate(student, val_loader)
        print(f"[{epoch}/{EPOCHS}] Loss: {sum_loss/cnt:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), 'student_best_cnn_lstm.pth')
            print(f"💾 Best model saved @ epoch={epoch}, acc={best_acc:.4f}")

    print(f"\n🎯 최종 검증 정확도: {best_acc:.4f}")
