import os
import json
import pandas as pd
import glob

# 데이터 폴더
json_dir = "C:/data/json1"  # JSON 파일 경로
image_dir = "C:/data/image1"  # 이미지 파일 경로

# 파일명에서 상태 추출 및 라벨 매핑
def get_label_from_filename(filename):
    if "하품" in filename:
        return 0
    elif "졸음" in filename:
        return 1
    else:
        return 2  # 정상주시, 흡연상태, 통화재현

data_list = []

# JSON 파일 순회하며 데이터 추출
json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
print(f"📂 Found {len(json_files)} JSON files")

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    image_name = data.get("FileInfo", {}).get("FileName")  # JSON에 저장된 이미지 파일명
    relative_path = os.path.relpath(json_file, json_dir)  # JSON 경로 상대 경로 변환
    image_path = os.path.join(image_dir, relative_path).replace("\\", "/")  # json -> image 변경
    image_path = image_path.rsplit(".", 1)[0] + ".jpg"  # 확장자 변경
        
    exists = os.path.exists(image_path)  # 이미지 존재 여부 확인

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

        # 라벨 추출
    label = get_label_from_filename(file_name)

        # 데이터 저장
    data_list.append([file_name, image_path, left_eye_open, right_eye_open, mouth_open, face_width, face_height, label])

print(f"✅ Loaded {len(data_list)} items")

# Pandas DataFrame 생성 및 CSV 저장
df = pd.DataFrame(data_list, columns=["FileName", "ImagePath", "LeftEye_Open", "RightEye_Open", "Mouth_Open", "Face_Width", "Face_Height", "Label"])
df.to_csv("drowsiness_dataset.csv", index=False)
print("CSV 저장 완료!")


