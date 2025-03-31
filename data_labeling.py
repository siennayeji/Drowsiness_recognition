import os
import json
from glob import glob
from PIL import Image
from torchvision import transforms
import torch

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, json_root_dir, image_root_dir, label_map, transform=None, action2idx=None, use_face=True):
        self.samples = []
        self.transform = transform
        self.label_map = label_map
        self.action2idx = action2idx
        self.use_face = use_face

        json_files = glob(os.path.join(json_root_dir, "**", "*.json"), recursive=True)
        print(f"🔍 총 JSON 파일 수: {len(json_files)}")

        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                category = data.get("scene_info", {}).get("category_name")
                if category not in label_map:
                    continue

                label = label_map[category]
                frames = data.get("scene", {}).get("data", [])
                # 프레임이 꼭 5개여야 하는 로직 그대로 유지
                if len(frames) != 5:
                    continue

                # JSON 경로 → image 폴더 내부 경로 계산
                rel_path = os.path.relpath(os.path.dirname(json_path), json_root_dir)
                rel_path = rel_path.replace("label", "").rstrip("/\\")
                
                # lazy loading을 위해 "이미지 경로와 bbox만" 저장
                img_paths = []
                bboxes = []
                actions = []

                valid_frames = True
                for frame in frames:
                    img_name = frame.get("img_name")
                    img_path = os.path.join(image_root_dir, rel_path, "img", img_name)
                    if not os.path.exists(img_path):
                        valid_frames = False
                        break

                    occupant = frame.get("occupant", [])
                    if not occupant:
                        valid_frames = False
                        break
                    
                    occupant_0 = occupant[0]
                    bbox = occupant_0.get("face_b_box") if self.use_face else occupant_0.get("body_b_box")
                    if not bbox or len(bbox) != 4:
                        valid_frames = False
                        break

                    img_paths.append(img_path)
                    bboxes.append(bbox)
                    actions.append(occupant_0.get("action", "기타"))

                if not valid_frames:
                    continue

                # 5장 모두 유효하면, 가장 많이 나온 action 하나를 top_action으로
                action_counts = {a: actions.count(a) for a in actions}
                top_action = max(action_counts, key=action_counts.get)

                # **중요**: 여기선 이미지 열지 않고, 필요한 정보만 저장
                self.samples.append({
                    "img_paths": img_paths,     # [str, str, str, str, str]
                    "bboxes": bboxes,          # [(x,y,w,h), (x,y,w,h), ...]
                    "label": label,
                    "top_action": top_action
                })

            except Exception as e:
                print(f"⚠️ {json_path} 처리 중 오류: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """ 실제로 이미지 파일 열고, crop & transform 적용 (lazy loading) """
        sample = self.samples[idx]
        img_paths = sample["img_paths"]
        
        
        s = sample["bboxes"]
        label = sample["label"]
        top_action = sample["top_action"]

        seq_imgs = []
        for (img_path, bbox) in zip(img_paths, sample["bboxes"]):
            x, y, w, h = bbox

            try:
                with Image.open(img_path).convert('RGB') as img:
                    crop = img.crop((x, y, x + w, y + h))
                    if self.transform:
                        crop = self.transform(crop)
                    seq_imgs.append(crop)
            except:
                # 만약 여기서 이미지 깨짐 등의 오류가 있으면 None 리턴 or 예외 처리
                # 보통은 이 샘플을 건너뛰거나, dummy tensor로 대체할 수도 있음
                # 예시로 그냥 오류 발생시키기:
                return None

        # seq_imgs를 하나의 텐서로 묶어서 반환
        seq_imgs = torch.stack(seq_imgs)  # shape: [5, C, H, W]
        return seq_imgs, label, top_action
