import os
import glob
import json
import torch
from PIL import Image
import subprocess

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, json_root_dir, image_root_dir, label_map, transform=None, use_face=True):
        self.samples = []
        self.transform = transform
        self.label_map = label_map
        self.use_face = use_face

        # ✅ find 명령어로 JSON 경로 수집 (수정됨)
        json_files = self.find_json_files(json_root_dir)
        print(f"🔍 [find] JSON 파일 수: {len(json_files)}")

        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                category = data.get("scene_info", {}).get("category_name")
                if category not in label_map:
                    continue

                label = label_map[category]
                frames = data.get("scene", {}).get("data", [])
                if len(frames) != 5:
                    continue

                # ✅ rel_path 수정 (중간에 'label'만 제거)
                rel_path = os.path.relpath(os.path.dirname(json_path), json_root_dir)
                rel_parts = rel_path.split(os.sep)
                if "label" in rel_parts:
                    rel_parts.remove("label")
                rel_path = os.path.join(*rel_parts)

                img_paths = []
                bboxes = []
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

                    bbox = occupant[0].get("face_b_box") if self.use_face else occupant[0].get("body_b_box")
                    if not bbox or len(bbox) != 4:
                        valid_frames = False
                        break

                    img_paths.append(img_path)
                    bboxes.append(bbox)

                if valid_frames:
                    self.samples.append({
                        "img_paths": img_paths,
                        "bboxes": bboxes,
                        "label": label
                    })

            except Exception as e:
                print(f"⚠️ JSON 처리 실패: {json_path} → {e}")

    def find_json_files(self, root):
        try:
            # ✅ 여기 수정: 'json' 붙이지 않음
            result = subprocess.run(["find", root, "-name", "*.json"],
                                    stdout=subprocess.PIPE, text=True, check=True)
            json_paths = result.stdout.strip().split("\n")
            return [p for p in json_paths if p.strip().endswith(".json")]
        except subprocess.CalledProcessError as e:
            print(f"❌ find 명령어 실패: {e}")
            return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_paths = sample["img_paths"]
        label = sample["label"]

        seq_imgs = []
        for img_path, bbox in zip(img_paths, sample["bboxes"]):
            x, y, w, h = bbox
            try:
                with Image.open(img_path).convert('RGB') as img:
                    crop = img.crop((x, y, x + w, y + h))
                    if self.transform:
                        crop = self.transform(crop)
                    seq_imgs.append(crop)
            except:
                return None

        seq_imgs = torch.stack(seq_imgs)
        return seq_imgs, label
