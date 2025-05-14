import os

path1 = "/home/sienna/my_ws/data"
path2 = "/home/sienna/sienna_data/json"

print("✅ my_ws 접근 가능?", os.path.exists(path1))
print("✅ sienna_data 접근 가능?", os.path.exists(path2))

for root, dirs, files in os.walk(path2):
    print("📂 탐색 중 디렉토리:", root)
    break  # 하나만 보기
