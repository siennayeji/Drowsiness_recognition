# === Knowledge Distillation (Teacher-Student) 모델 학습 예제 ===
import os
import json
from glob import glob
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms, models

# --- 하이퍼파라미터 설정 ---
BATCH_SIZE = 8
EPOCHS = 10
SEQ_LEN = 5
NUM_CLASSES = 2
LR = 1e-4
TEMPERATURE = 4.0
ALPHA = 0.7  # KD와 CE의 가중치

# --- 라벨 매핑 ---
label_map = {
    "졸음운전": 1,
    "음주운전": 0,
    "물건찾기": 0,
    "통화": 0,
    "휴대폰조작": 0,
    "차량제어": 0,
    "운전자폭행": 0
}

# --- 행동 벡터 정의 ---
action_vocab = ["하품", "꾸벅꾸벅졸다", "눈비비기", "눈깜빡이기", "전방주시", "운전하다", "기타"]
action2idx = {a: i for i, a in enumerate(action_vocab)}
ACTION_DIM = len(action_vocab)

# --- 데이터 전처리 ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Dataset 정의 (얼굴 크롭) ===
class SequenceDataset(Dataset):
    def __init__(self, json_root, img_root, label_map, transform, use_face=True):
        self.samples = []
        self.transform = transform
        self.label_map = label_map
        self.use_face = use_face
        files = glob(os.path.join(json_root, '**', '*.json'), recursive=True)
        for j in files:
            try:
                data = json.load(open(j, 'r', encoding='utf-8'))
                cat = data.get('scene_info', {}).get('category_name')
                if cat not in label_map:
                    continue
                frames = data.get('scene', {}).get('data', [])
                if len(frames) != SEQ_LEN:
                    continue
                base = os.path.relpath(os.path.dirname(j), json_root).replace('label', '').rstrip('/\\')
                paths, boxes, ok = [], [], True
                for fr in frames:
                    p = os.path.join(img_root, base, 'img', fr['img_name'])
                    if not os.path.exists(p):
                        ok = False
                        break
                    occ = fr.get('occupant', [])
                    if not occ:
                        ok = False
                        break
                    bbox = occ[0].get('face_b_box')
                    if not bbox or len(bbox) != 4:
                        ok = False
                        break
                    paths.append(p)
                    boxes.append(bbox)
                if ok:
                    self.samples.append((paths, boxes, label_map[cat]))
            except:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, boxes, label = self.samples[idx]
        seq = []
        for p, box in zip(paths, boxes):
            x, y, w, h = box
            img = Image.open(p).convert('RGB').crop((x, y, x + w, y + h))
            seq.append(self.transform(img))
        return torch.stack(seq), label

# === 원본 모델과 동일한 CNNEncoder 정의 ===
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

# === Teacher 모델 (액션 벡터 포함) ===
class TeacherModel(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # 원본 학습 코드와 동일한 encoder 사용
        self.encoder = CNNEncoder()
        self.lstm    = nn.LSTM(64, 128, batch_first=True)
        self.fc      = nn.Linear(128 + action_dim, NUM_CLASSES)

    def forward(self, x, a):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x).view(B, T, -1)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        return self.fc(torch.cat([last, a], dim=1))

# === Student 모델 (얼굴만) ===
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2], nn.AdaptiveAvgPool2d(1))
        self.lstm    = nn.LSTM(512, 128, batch_first=True)
        self.fc      = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x).view(B, T, -1)
        out, _ = self.lstm(feats)
        return self.fc(out[:, -1, :])

# === 메인 ===
if __name__ == '__main__':
    # 1) 데이터 로드 및 통계
    json_dir = '/home/sienna/sienna_data/json'
    img_dir  = '/home/sienna/sienna_data/image'
    full_ds  = SequenceDataset(json_dir, img_dir, label_map, train_transform)
    print(f"✅ 전체 데이터셋 길이: {len(full_ds)}")

    # 2) 라벨 분포 계산
    filtered_ds = Subset(full_ds, list(range(len(full_ds))))
    labels = []
    for i in range(len(filtered_ds)):
        try:
            seq, lbl = filtered_ds[i]
            labels.append(lbl)
        except:
            continue
    counter = Counter(labels)
    print(f"📊 라벨 분포: {counter}")

    # 3) Train/Val 분할
    n_train = int(0.8 * len(filtered_ds))
    n_val   = len(filtered_ds) - n_train
    train_ds, val_ds = random_split(filtered_ds, [n_train, n_val])

    # 4) 불균형 보정: WeightedRandomSampler
    train_labels = [full_ds.samples[idx][2] for idx in train_ds.indices]
    weights = [1.0 / counter[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=lambda b: default_collate([x for x in b if x is not None])
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: default_collate([x for x in b if x is not None])
    )

    # 5) Teacher 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEACHER_PATH = '/home/sienna/my_ws/code/cnn_lstm_drowsiness_1.pth'
    teacher = TeacherModel(action_dim=ACTION_DIM).to(device).eval()
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    for p in teacher.parameters():
        p.requires_grad = False

    # 6) Student 모델 학습 준비
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
                total += y.size(0)
        return correct / total if total else 0

    # 7) Knowledge Distillation 학습 & 베스트 저장
    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        student.train()
        sum_loss, cnt = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # action vector (실제로 사용할 데이터가 없으면 0으로 채움)
            a = torch.zeros(x.size(0), ACTION_DIM, device=device)
            with torch.no_grad():
                t_logits = teacher(x, a)
            s_logits = student(x)
            t_soft = nn.functional.softmax(t_logits / TEMPERATURE, dim=1)
            s_logp = nn.functional.log_softmax(s_logits / TEMPERATURE, dim=1)
            loss_kd = kd_loss(s_logp, t_soft) * (TEMPERATURE ** 2)
            loss_ce = ce_loss(s_logits, y)
            loss = ALPHA * loss_kd + (1 - ALPHA) * loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            cnt += 1

        avg_loss = sum_loss / cnt
        val_acc = evaluate(student, val_loader)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), 'student_best.pth')
            print(f"💾 [Best] epoch={epoch}, acc={best_acc:.4f}")

    print(f"\n🎯 최종 검증 정확도: {best_acc:.4f}")
