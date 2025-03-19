import os
import json
import pandas as pd
import glob

# 🔹 데이터 폴더 (Train / Validation 따로 지정)
train_json_dir = "C:/data/train/json"  # Train JSON 경로
train_image_dir = "C:/data/train/image"  # Train 이미지 경로

val_json_dir = "C:/data/val/json"  # Validation JSON 경로
val_image_dir = "C:/data/val/image"  # Validation 이미지 경로

# 🔹 파일명에서 상태 추출 및 라벨 매핑
def get_label_from_filename(filename):
    if "하품" in filename:
        return 0
    elif "졸음" in filename:
        return 1
    else:
        return 2  # 정상주시, 흡연상태, 통화재현

# 🔹 JSON 파일을 읽고 CSV로 변환하는 함수
def process_json_files(json_dir, image_dir):
    data_list = []
    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
    print(f"📂 Found {len(json_files)} JSON files in {json_dir}")

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_name = data.get("FileInfo", {}).get("FileName")  # JSON에 저장된 이미지 파일명
        relative_path = os.path.relpath(json_file, json_dir)  # JSON 경로 상대 경로 변환
        image_path = os.path.join(image_dir, relative_path).replace("\\", "/")  # json -> image 변경
        image_path = image_path.rsplit(".", 1)[0] + ".jpg"  # 확장자 변경

        if not os.path.exists(image_path):
            continue  # 이미지 파일이 없으면 건너뜀

        file_name = data["FileInfo"]["FileName"]
        label = get_label_from_filename(file_name)

        # 눈과 입 상태 확인
        left_eye_open = int(data["ObjectInfo"]["BoundingBox"]["Leye"]["Opened"])
        right_eye_open = int(data["ObjectInfo"]["BoundingBox"]["Reye"]["Opened"])
        mouth_open = int(data["ObjectInfo"]["BoundingBox"]["Mouth"]["Opened"])

        # 얼굴 크기 계산
        face_box = data["ObjectInfo"]["BoundingBox"]["Face"]["Position"]
        face_width = face_box[2] - face_box[0]
        face_height = face_box[3] - face_box[1]

        # 데이터 저장
        data_list.append([file_name, image_path, left_eye_open, right_eye_open, mouth_open, face_width, face_height, label])

    print(f"✅ Loaded {len(data_list)} items from {json_dir}")
    return data_list

# 🔹 Train 데이터 처리
train_data = process_json_files(train_json_dir, train_image_dir)
df_train = pd.DataFrame(train_data, columns=["FileName", "ImagePath", "LeftEye_Open", "RightEye_Open", "Mouth_Open", "Face_Width", "Face_Height", "Label"])
df_train.to_csv("train.csv", index=False)
print("✅ Train CSV 저장 완료!")

# 🔹 Validation 데이터 처리
val_data = process_json_files(val_json_dir, val_image_dir)
df_val = pd.DataFrame(val_data, columns=["FileName", "ImagePath", "LeftEye_Open", "RightEye_Open", "Mouth_Open", "Face_Width", "Face_Height", "Label"])
df_val.to_csv("val.csv", index=False)
print("✅ Validation CSV 저장 완료!")
