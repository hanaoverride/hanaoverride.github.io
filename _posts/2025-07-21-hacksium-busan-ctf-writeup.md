---
layout: post
title: "HACKSIUM BUSAN 2025 본선 마지막날 문제 중 서명 복원문제 풀이"
date: 2025-07-21 17:03:31 +0900
categories: [security]
tags: [HACKSIUM, CTF, ModelInversion, DeepLearning, WriteUp]
redirect_from:
  - /security/2025/07/21/hacksium-busan-ctf-writeup/
---

문제 해결 목표는 **4개의 유효한 서명 이미지를 생성하여 서버에 업로드하고, 서버의 검증 로직을 통과하여 최종적으로 플래그를 획득하는 것**입니다.

### 취약점 분석 및 문제 해결 전략

**1. 전자 서명 검증 시스템의 목적과 한계**

전자 서명 시스템의 목적은 제출된 서명이 미리 등록된 원본 서명과 일치하는지 판별하여 사용자의 신원을 확인하는 것입니다. 일반적으로 딥러닝 모델은 이미지의 특징(feature)을 추출하여 이 유사성을 비교하는 데 사용될 수 있습니다. 하지만 이 과정에서 사용되는 모델의 구조나 목표값(원본 서명의 특징 벡터)이 노출될 경우, 공격자는 이 정보를 역으로 이용하여 검증을 통과하는 위조 이미지를 생성할 수 있습니다.

**2. 취약점 (Model Inversion Vulnerability)**

본 문제의 핵심 취약점은 **검증 시스템의 내부 로직과 데이터가 모두 공격자에게 노출되어 있다는 점**입니다. `main.py` 코드와 제공된 파일을 통해 다음과 같은 정보들을 알 수 있으며, 이는 모델 역산 공격(Model Inversion Attack)을 가능하게 합니다.

- **검증 로직의 완전한 공개**: `main.py`의 `verify_signature` 함수는 이미지 전처리 과정부터 모델을 통해 특징 벡터를 추출하고, 미리 저장된 특징 벡터와 MSE(평균 제곱 오차)를 비교하여 0.000001 (1e-6) 미만일 경우에만 유효하다고 판단하는 전체 로직을 명확하게 보여줍니다.
- **검증 모델 및 가중치 파일 제공**: `SignatureCNN` 클래스로 정의된 모델의 구조와 `model/model.pth`로 제공된 학습된 가중치(weights)를 통해 공격자는 서버와 동일한 모델을 로컬 환경에서 완벽하게 재현할 수 있습니다.
- **목표 특징 벡터 제공**: 각 서명(`signature1`~`signature4`)에 대한 원본 특징 벡터가 `model/mse_loss_mat_{index}.bin` 파일에 담겨 제공됩니다. 이는 공격의 "정답지" 역할을 하며, 위조 이미지가 도달해야 할 명확한 목표를 제시합니다.
- **서버 측 이미지 처리 방식 노출**: `Image.open(io.BytesIO(image_data)).convert('L').resize((28, 28))` 코드를 통해 서버가 이미지를 어떻게 처리하는지 정확히 알 수 있습니다. 공격 성공을 위해서는 이 과정을 로컬에서 동일하게 모방하여 손실(loss)을 계산해야 합니다.

### 문제 해결 전략

이러한 취약점을 바탕으로, 모델 역산 공격을 통해 서버의 검증을 통과할 수 있는 서명 이미지를 생성하는 전략을 수립합니다.

- **1. 환경 구성**: 서버와 동일한 `SignatureCNN` 모델을 로드하고, 제공된 `model.pth` 가중치를 적용합니다. 또한, 4개의 `mse_loss_mat_*.bin` 파일로부터 목표 특징 벡터를 로드합니다.
- **2. 이미지 초기화**: 임의의 노이즈로 채워진 28x28 크기의 이미지를 생성합니다. 이 이미지가 최적화의 시작점입니다.
- **3. 손실 함수 정의**: 손실(Loss)은 **현재 생성 중인 이미지에서 추출한 특징 벡터**와 **목표 특징 벡터** 간의 **MSE(평균 제곱 오차)**로 정의합니다. 우리의 목표는 이 손실 값을 1e-6 미만으로 최소화하는 것입니다.
- **4. 이미지 최적화 (역산 수행)**:
  - Adam과 같은 최적화 알고리즘(Optimizer)을 사용하여 손실 함수를 최소화하는 방향으로 이미지의 각 픽셀 값을 반복적으로 업데이트합니다.
  - 이 과정은 모델을 학습시키는 것이 아니라, **고정된 모델을 기준으로 입력 이미지를 정답에 가깝게 "깎아나가는"** 과정입니다.
- **5. 검증 로직 시뮬레이션 및 하이퍼파라미터 탐색**:
  - 서버의 검증 과정을 로컬에서 정확히 시뮬레이션(`verify_locally` 함수)하여 현재 생성된 이미지가 실제로 서버에서 통과될 수 있는지 주기적으로 확인합니다.
  - 최적화 과정이 지역 최솟값(local minima)에 빠져 더 이상 손실이 줄어들지 않는 경우를 대비해, 여러 하이퍼파라미터(학습률, Adam 베타 값 등) 조합과 다른 랜덤 시드(시작 노이즈)로 여러 번의 재시도를 수행하는 탐색 공간(Search Space)을 정의하여 성공 확률을 높입니다.
- **6. 성공 및 저장**: 시뮬레이션된 손실 값이 목표치(안전하게 9e-7로 설정) 미만으로 떨어지면 성공으로 간주하고, 생성된 이미지를 PNG 파일로 저장합니다. 이 과정을 4개의 모든 서명에 대해 반복합니다.

### 솔루션: 모델 역산을 통한 서명 이미지 복원

아래 코드는 위 전략을 구현한 PoC(Proof of Concept)입니다. 하이퍼파라미터 탐색 공간을 정의하고, 각 이미지에 대해 성공할 때까지 여러 전략과 랜덤 시드로 재시도하여 안정적으로 위조 서명을 생성합니다.

```python
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import io

# 서버의 성공 기준(1e-6)보다 약간 더 엄격한 목표 손실 값 설정
TARGET_FEATURE_LOSS = 9e-7

# 서버의 모델 아키텍처와 동일하게 정의
class SignatureCNN(nn.Module):
  def __init__(self):
    super(SignatureCNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.fc1 = nn.Linear(128 * 3 * 3, 1024)

    self.relu = nn.ReLU()
    self.max_pool = nn.MaxPool2d(2, 2)
  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.max_pool(x)
    x = self.relu(self.conv2(x))
    x = self.max_pool(x)
    x = self.relu(self.conv3(x))
    x = self.max_pool(x)
    x = x.view(-1, 128 * 3 * 3)
    x = self.fc1(x)
    return x

def total_variation_loss(img):
  """생성된 이미지의 노이즈를 줄여 더 부드럽게 만드는 정규화 함수 (본 풀이에서는 사용되지 않음)"""
  bs_img, c_img, h_img, w_img = img.size()
  tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
  tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
  return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

def verify_locally(model, image_tensor, reference_features_server):
  """
  서버의 검증 로직을 그대로 시뮬레이션하는 함수.
  PyTorch 텐서를 PNG로 변환했다가 다시 PIL로 읽어오는 과정을 거쳐
  서버에서 발생할 수 있는 미세한 손실 변화까지 정확하게 계산한다.
  """
  with io.BytesIO() as buffer:
    transforms.ToPILImage()(image_tensor.squeeze(0)).save(buffer, format='PNG')
    # 서버와 동일한 전처리: Bytes -> PIL Image -> Grayscale -> Resize -> Tensor
    tensor_from_png = transforms.ToTensor()(Image.open(io.BytesIO(buffer.getvalue())).convert('L').resize((28, 28))).unsqueeze(0)
  with torch.no_grad():
    features = model(tensor_from_png)
  return torch.nn.functional.mse_loss(features, reference_features_server).item()

def reconstruct_signature_adam_only(model, reference_features_server, output_path,
                  iterations, lr, tv_weight,
                  scheduler_patience, scheduler_factor, adam_betas, random_seed):
  """주어진 하이퍼파라미터로 단일 이미지 복원을 시도하는 함수"""
  print(f"[*] Reconstructing {output_path} (Seed: {random_seed})...")

  torch.manual_seed(random_seed) # 다른 시작점을 위해 랜덤 시드 설정
  # 최적화를 시작할 랜덤 노이즈 이미지 생성
  reconstructed_image = torch.randn(1, 1, 28, 28, requires_grad=True)
  optimizer = torch.optim.Adam([reconstructed_image], lr=lr, betas=adam_betas) # Adam 옵티마이저 사용
  # 학습률 스케줄러: 손실이 정체될 때 학습률을 동적으로 감소시켜 미세 조정
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                             factor=scheduler_factor,
                             patience=scheduler_patience)

  for i in range(iterations):
    optimizer.zero_grad()
    features = model(reconstructed_image)
    feature_loss = torch.nn.functional.mse_loss(features, reference_features_server)
    total_loss = feature_loss + tv_weight * total_variation_loss(reconstructed_image) # TV Loss는 가중치가 0
    total_loss.backward() # 역전파로 그래디언트 계산
    optimizer.step()      # 이미지 픽셀 업데이트
    scheduler.step(total_loss.item()) # 스케줄러 업데이트

    with torch.no_grad():
      reconstructed_image.clamp_(0, 1) # 픽셀 값을 0과 1 사이로 유지

    if (i + 1) % 2000 == 0:
      simulated_loss = verify_locally(model, reconstructed_image.detach(), reference_features_server)
      print(f"  - Iter [{i+1}/{iterations}], Simulated Loss: {simulated_loss:.10f}")
      if simulated_loss < TARGET_FEATURE_LOSS:
        break # 목표 도달 시 조기 종료

  final_simulated_loss = verify_locally(model, reconstructed_image.detach(), reference_features_server)
  print(f"  - Final Simulated Server Loss: {final_simulated_loss:.10f}")

  if final_simulated_loss < TARGET_FEATURE_LOSS:
    print("  - SUCCESS: Target reached. Saving final image.")
    transforms.ToPILImage()(reconstructed_image.squeeze(0)).save(output_path)

  return final_simulated_loss

if __name__ == "__main__":
  model = SignatureCNN()
  model.load_state_dict(torch.load('model/model.pth', map_location=torch.device('cpu')))
  model.eval()

  # [핵심] 성공률을 높이기 위한 하이퍼파라미터 탐색 공간 정의
  # 각 시도는 기본, 공격적, 베타 수정 3가지 하위 전략을 가짐
  search_space = [
    # 1차 시도: 일반적인 설정
    {'name': 'Standard', 'tv_weight': 0.0, 'iterations': 25000, 'lr': 0.01, 'patience': 500, 'factor': 0.2, 'betas': (0.9, 0.999)},
    # 2차 시도: 더 긴 반복과 민감한 스케줄러로 정밀 탐색
    {'name': 'Aggressive', 'tv_weight': 0.0, 'iterations': 40000, 'lr': 0.01, 'patience': 300, 'factor': 0.2, 'betas': (0.9, 0.999)},
    # 3차 시도: Adam의 관성(beta1)을 줄여 지역 최솟값 탈출 시도
    {'name': 'Beta Tweak', 'tv_weight': 0.0, 'iterations': 40000, 'lr': 0.01, 'patience': 300, 'factor': 0.2, 'betas': (0.8, 0.999)},
  ]

  max_retries_per_image = 5 # 한 전략 내에서 최대 5번의 다른 랜덤 시드로 재시도
  final_results = {}

  for img_idx in range(4): # 4개의 서명에 대해 반복
    print(f"\n{'='*70}\n--- Processing Image #{img_idx} ---\n{'='*70}")

    image_solved = False
    # 정의된 전략들을 순서대로 시도
    for attempt_num, params in enumerate(search_space):
      if image_solved: break

      print(f"\n>>> Attempt #{attempt_num+1} with Strategy: {params['name']}")

      # 한 전략이 실패하면 다른 랜덤 시드로 재시도
      for retry_num in range(max_retries_per_image):
        random_seed = int.from_bytes(os.urandom(4), 'big') # 매번 새로운 랜덤 시드 생성

        mse_file_path = f'model/mse_loss_mat_{img_idx}.bin'
        output_image_path = f'recon_img_{img_idx}.png'
        reference_features_server = torch.tensor(np.matrix(np.fromfile(mse_file_path, dtype=np.float32).reshape(1, 1024)), dtype=torch.float32)

        simulated_server_loss = reconstruct_signature_adam_only(
          model, reference_features_server, output_image_path,
          iterations=params['iterations'], lr=params['lr'], tv_weight=params['tv_weight'],
          scheduler_patience=params['patience'], scheduler_factor=params['factor'],
          adam_betas=params['betas'], random_seed=random_seed
        )

        if simulated_server_loss < TARGET_FEATURE_LOSS:
          print(f"\n[SUCCESS] Image #{img_idx} SOLVED with strategy '{params['name']}' on retry #{retry_num+1}!")
          final_results[img_idx] = {'status': 'Success', 'loss': simulated_server_loss, 'params': params['name']}
          image_solved = True
          break # 현재 이미지에 대한 모든 시도 중단하고 다음 이미지로 이동

    if not image_solved:
      print(f"\n[FAILURE] Image #{img_idx} could NOT be solved after all attempts.")
      final_results[img_idx] = {'status': 'Failure'}

  print(f"\n\n{'='*70}\n--- FINAL REPORT ---\n{'='*70}")
  all_solved = all(res.get('status') == 'Success' for res in final_results.values())
  for idx, result in final_results.items():
    print(f"Image {idx}: {result}")

  if all_solved:
    print("\n[CONGRATULATIONS] All images have been successfully reconstructed!")
  else:
    print("\n[SEARCH COMPLETE] Some images could not be solved.")
```

### 관련 개념

- **모델 역산 공격 (Model Inversion Attack)**: 머신러닝 모델에 대한 접근 권한(주로 API 형태)을 이용하여 모델이 학습한 데이터의 민감한 정보를 복원하려는 공격 기법입니다. 본 문제에서는 모델 자체와 목표값까지 주어져 공격이 더욱 용이합니다.
- **특징 벡터 (Feature Vector)**: 딥러닝 모델(특히 CNN)이 입력 이미지에서 핵심적인 패턴과 정보를 추출하여 표현하는 고차원의 숫자 배열입니다. `SignatureCNN`의 마지막 레이어에서 나오는 1024차원 벡터가 이에 해당합니다.
- **평균 제곱 오차 (Mean Squared Error, MSE)**: 예측값과 실제값 사이의 차이를 제곱하여 평균을 낸 값으로, 두 벡터가 얼마나 유사한지를 측정하는 손실 함수입니다. 이 값이 작을수록 두 벡터는 더 가깝습니다.
- **경사 하강법 (Gradient Descent) / Adam Optimizer**: 손실 함수의 기울기(gradient)를 계산하여 손실이 최소화되는 방향으로 모델의 파라미터(또는 본 문제에서는 입력 이미지의 픽셀)를 점진적으로 업데이트하는 최적화 알고리즘입니다. Adam은 이 과정에서 속도와 안정성을 개선한 고급 버전입니다.
- **하이퍼파라미터 (Hyperparameter)**: 모델 학습이나 최적화 과정에 영향을 주는 설정값으로, 개발자가 직접 지정해야 합니다. 학습률(learning rate), 반복 횟수(iterations), Adam의 베타 값 등이 있으며, 이 값들을 어떻게 설정하느냐에 따라 최적화의 성공 여부가 크게 달라집니다.
- **지역 최솟값 (Local Minima)**: 최적화 과정에서 전체적인 최솟값이 아닌, 특정 구간 내에서의 최솟값에 도달하여 더 이상 개선이 이루어지지 않는 상태를 의미합니다. 다른 랜덤 시드(시작점)나 하이퍼파라미터로 재시도하는 것은 이 상태를 벗어나기 위한 효과적인 전략입니다.
- **Total Variation (TV) Loss)**: 생성된 이미지의 인접 픽셀 간 차이를 줄여 이미지를 부드럽게 만들고 노이즈를 억제하는 데 사용되는 정규화(regularization) 기법입니다. 본 PoC에서는 최종적으로 사용하지 않았지만(가중치 0), 이미지 생성 분야에서 흔히 사용됩니다.
