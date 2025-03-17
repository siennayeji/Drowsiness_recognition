import torch
print(torch.cuda.is_available())  # True여야 GPU 사용 가능
print(torch.cuda.device_count())  # 사용 가능한 GPU 개수
print(torch.cuda.get_device_name(0))  # 사용 중인 GPU 이름
