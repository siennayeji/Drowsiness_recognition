import json

json_path = "/home/sienna/sienna_data/092.한국인_감정인식을_위한_복합_영상_데이터/01.데이터/1.Training/라벨링데이터/슬픔_unzipped/img_emotion_training_data(슬픔).json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 앞에 5개 filename만 출력
for i in range(5):
    print(data[i]["filename"])
