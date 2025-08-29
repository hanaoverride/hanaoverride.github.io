---
layout: post
title: "Transformer 스터디 (3) Feed Forward Network: 참으로 직관적인 이름이네!"
date: 2025-08-29 10:10:03 +0900
categories: [llm-engineering]
tags: [Transformer, FeedForward, FFN, 비선형성, 딥러닝]
redirect_from:
  - /llm-engineering/2025/08/29/transformer-study-part3/
---
# 정보를 변환하는 엔진 - Feed Forward Network의 역할

> *"Attention만으로는 뭔가 부족해 보이는데, FFN은 정확히 뭘 하는 거지?"*

## 지난 이야기와 새로운 의문

지금까지 Transformer의 두 핵심 요소를 알아봤죠. Self-Attention으로 단어 간 관계를 파악하고, Positional Encoding으로 순서 정보를 주입하는 방법. 하지만 스터디 3주차에 현우가 던진 질문이 새로운 궁금증을 불러일으켰어요.

**"그런데 Attention이 관계만 찾아주면 끝인가? 그 다음엔 뭐가 일어나지?"**

실제로 Transformer 구조를 보면 Self-Attention 다음에 꼭 등장하는 게 있어요. **Feed Forward Network (FFN)**. 이게 정확히 뭘 하는 걸까요?

## Feed Forward Network란 무엇인가?

우선 문장부터 무슨 뜻인지 뜯어볼까요?
"Feed Forward Network"를 단어별로 그대로 해체해서 보니까 더 직관적으로 다가왔어요.

> **현우**: "feed가 '먹이를 주다' 말고도 '무언가를 입력으로 밀어 넣다'는 느낌이잖아? forward는 '앞으로만 간다'는 거고, network는 '연결된 연산 구조'고... 그러면 그냥 '입력이 뒤로 새지 않고 앞으로 한 번 쭉 흐르며 변환되는 층 묶음'이라는 뜻 아닌가?"
>
> **지영**: "맞아. RNN처럼 되돌아가거나 순환(recurrent)하지 않고, CNN처럼 공간적 구조(convolution)를 미는 게 아니라, 그냥 입력을 받아 비선형 한 번 주고 다시 선형 변환해서 다른 표현 공간으로 '전달(feed)'하는 거네."
>
> **민수**: "그래서 '피드-포워드'라는 이름 자체가 구조를 설명하는 매뉴얼이었네. 복잡한 게 아니라 너무 직설적이라 놓쳤던 거구나."

이렇게 단어를 그대로 뜯어보니: (1) 순환 없음, (2) 되돌아가는 경로 없음, (3) 각 입력이 독립적으로 처리, (4) 주된 목적은 표현 변환이라는 사실이 동시에 정리되면서 역할이 확 다가왔죠.

### 가장 기본적인 형태

FFN은 사실 우리가 아는 가장 일반적인 신경망이에요:

```python
# 의사코드
def feed_forward_network(x):
    # 첫 번째 선형변환 + ReLU
    hidden = ReLU(W1 @ x + b1)
    
    # 두 번째 선형변환  
    output = W2 @ hidden + b2
    
    return output
```

**스터디에서 첫 반응:**

> **민수**: "이거 그냥 일반적인 MLP(Multi-Layer Perceptron) 아닌가?"
> 
> **지영**: "맞아, 이게 뭐 특별한 건지 모르겠는데..."
> 
> **현우**: "Attention이 복잡한데 이건 너무 단순한 거 아니야?"

바로 그 "단순함"이 핵심이었어요.

### Position-wise의 의미

FFN의 정식 명칭은 "Position-wise Feed Forward Network"예요. 이 "Position-wise"가 중요한 포인트죠.

```python
# 의사코드
# Position-wise 처리의 의미
sequence = [token1, token2, token3, token4]

# 각 토큰이 독립적으로 같은 FFN을 통과
for i, token in enumerate(sequence):
    processed_token = FFN(token)  # 동일한 FFN 적용
    sequence[i] = processed_token
```

**이때 나온 깨달음:**

> **지영**: "아! 각 위치의 토큰이 개별적으로 처리되는구나."
> 
> **현우**: "Attention은 토큰 간 상호작용이고, FFN은 각 토큰 내부 변환이네."
> 
> **민수**: "그럼 FFN은 토큰 하나하나를 '업그레이드'시키는 거야?"

정확한 이해였어요!

## 왜 Attention만으로는 부족할까?

### Attention의 한계

스터디에서 실험해본 결과, Attention만으로는 한계가 있었어요:

```python
# Self-Attention의 본질
# 가중평균으로 정보를 결합할 뿐, 새로운 정보를 생성하지는 못함
attention_output = weighted_sum_of_input_vectors

# 입력 벡터들의 선형결합일 뿐
# 입력에 없던 새로운 패턴이나 특성을 만들어낼 수 없음
```

**스터디 토론:**

> **현우**: "그럼 Attention은 정보를 '섞는' 역할만 하는 거네?"
> 
> **지영**: "맞아, 새로운 정보를 '만들어내지'는 못하고."
> 
> **민수**: "FFN이 그 역할을 하는 건가? 정보를 실제로 변환시키는?"

### 정보 변환의 필요성

언어 처리에서는 단순한 정보 결합을 넘어선 **변환**이 필요해요:

- "좋다"와 "훌륭하다"의 관계 학습
- 문맥에 따른 의미 변화 처리
- 추상적 개념 추출
- 복합적 의미 생성

이런 작업들은 비선형 변환이 필수죠.

## FFN의 실제 역할 - 단계별 추적

### 실험 설계

스터디에서 FFN의 각 단계가 어떤 변화를 일으키는지 추적해봤어요:

```python
import numpy as np

# 예시 입력 (어텐션 후 출력이라고 가정)
input_data = np.array([
    [1.0, -2.0, 0.5],   # 토큰 1
    [-0.5, 1.2, 0.0]    # 토큰 2
])

# FFN 파라미터 (간단한 예시)
W1 = np.array([[0.3, -1.2, 2.1], [1.1, 0.7, -0.8]])  # (2, 3)
b1 = np.array([0.1, -0.2])
W2 = np.array([[-0.4, 0.6], [0.8, -0.3], [0.2, 0.9]])  # (3, 2)  # hidden(2) -> output(3) so later transpose
b2 = np.array([0.05, -0.1, 0.15])

def relu(x):
    return np.maximum(0, x)

# 단계별 처리
print("=== 단계별 FFN 처리 ===")
print("입력:", input_data)

# 1단계: 첫 번째 선형변환
linear1_output = input_data @ W1.T + b1
print("1단계 (선형변환):", linear1_output)

# 2단계: ReLU 적용
relu_output = relu(linear1_output)  
print("2단계 (ReLU):", relu_output)

# 3단계: 두 번째 선형변환 (W2.T 사용하여 (2,2) -> (2,3))
final_output = relu_output @ W2.T + b2
print("최종 출력:", final_output)
```

### 실험 결과 분석

```
=== 단계별 FFN 처리 ===
입력: [[ 1.  -2.   0.5]
 [-0.5  1.2  0. ]]
1단계 (선형변환): [[ 3.85 -0.9 ]
 [-1.49  0.09]]
2단계 (ReLU): [[3.85 0.  ]
 [0.   0.09]]
최종 출력: [[-1.49   2.98   0.92 ]
 [ 0.104 -0.127  0.231]]
```

**이 결과를 보고 나온 반응:**

> **민수**: "ReLU에서 음수가 0으로 바뀌네. 정보가 손실되는 거 아냐?"
> 
> **지영**: "아니야, 이게 비선형성을 만드는 거야. 선형변환만으론 복잡한 패턴을 못 배우거든."
> 
> **현우**: "각 토큰이 독립적으로 변환되는 게 확실히 보이네."

## 비선형성의 중요성

### ReLU의 역할

ReLU 활성화 함수가 왜 중요한지 직접 확인해봤어요:

```python
# ReLU 없이 처리하면?
def linear_only_ffn(x):
    return W2 @ (W1 @ x + b1) + b2
    # = (W2 @ W1) @ x + (W2 @ b1 + b2)
    # = W_combined @ x + b_combined
    # 결국 하나의 선형변환과 동일!

# ReLU 있을 때
def nonlinear_ffn(x):
    hidden = relu(W1 @ x + b1)
    return W2 @ hidden + b2
    # 진정한 비선형 변환!
```

**핵심 깨달음:**

> **현우**: "ReLU가 없으면 아무리 레이어를 쌓아도 선형변환이구나."
> 
> **지영**: "비선형성이 있어야 복잡한 패턴을 학습할 수 있고."
> 
> **민수**: "그래서 언어의 복잡한 의미 관계를 처리할 수 있는 거네."

## FFN이 학습하는 패턴들

### 실제 학습 내용 (추론)

FFN이 실제로 어떤 것들을 학습하는지 연구 결과들을 보면:

1. **어휘적 관계**: "good" → "excellent" 같은 유의어 관계
2. **문법적 패턴**: 주어-동사 일치, 시제 변화 등
3. **의미적 조합**: 여러 특징을 조합한 복합 의미
4. **문맥적 조정**: 동일 단어의 문맥별 의미 변화

```python
# 개념적 예시 - FFN이 학습할 수 있는 패턴
def conceptual_ffn_patterns(token_embedding):
    # 1차 변환: 기본 특성 추출
    features = extract_basic_features(token_embedding)
    # [품사, 감정극성, 구체성, 시제, ...]
    
    # 비선형 활성화: 특성 조합
    activated_features = relu(features)
    # 특정 임계값 이상의 특성만 활성화
    
    # 2차 변환: 복합 의미 생성
    enhanced_embedding = combine_features(activated_features)
    # 문맥에 맞는 최종 임베딩 생성
    
    return enhanced_embedding
```

## 차원 확장의 비밀

### 왜 중간 차원을 늘릴까?

Transformer 논문에서 FFN의 중간 차원은 보통 입력의 4배예요 (d_model → 4*d_model → d_model).

**스터디에서 나온 의문:**

> **민수**: "왜 굳이 차원을 늘렸다가 다시 줄여?"
> 
> **지영**: "메모리만 더 쓰는 거 아닌가?"

### 표현력 확장의 원리

차원 확장의 이유를 실험으로 확인해봤어요:

```python
# 차원별 표현력 비교
def low_dim_ffn(x):  # 512 → 512 → 512
    return W2_low @ relu(W1_low @ x + b1_low) + b2_low

def high_dim_ffn(x):  # 512 → 2048 → 512  
    return W2_high @ relu(W1_high @ x + b1_high) + b2_high

# 고차원 중간층의 장점:
# 1. 더 많은 뉴런 = 더 복잡한 패턴 학습 가능
# 2. 더 세밀한 특성 분해
# 3. 정보 손실 없는 변환
```

**깨달음:**

> **현우**: "중간에 차원을 늘리면 더 복잡한 변환이 가능하구나."
> 
> **지영**: "마지막에 다시 줄여서 일관된 차원을 유지하고."
> 
> **민수**: "병목을 만들지 않으면서도 표현력을 확장하는 거네."

## Attention과 FFN의 협업

### 정보 처리 파이프라인

전체적인 처리 과정을 보면:

```python
def transformer_block(x):
    # 1. Self-Attention: 토큰 간 관계 파악
    attended = self_attention(x)
    x = x + attended  # Residual connection
    x = layer_norm(x)
    
    # 2. FFN: 각 토큰 개별 변환
    transformed = feed_forward(x)
    x = x + transformed  # Residual connection  
    x = layer_norm(x)
    
    return x
```

**역할 분담:**
- **Attention**: "어떤 정보를 가져올까?" (정보 수집)
- **FFN**: "이 정보를 어떻게 변환할까?" (정보 가공)

## 실제 언어 처리에서의 의미

### 구체적 예시

"The bank can guarantee deposits will eventually cover future tuition costs because it has enough money."

이 문장에서 "bank"의 의미 처리:

1. **Attention**: "deposits", "money"와의 관련성 파악
2. **FFN**: 금융기관으로서의 "bank" 의미로 변환 (강둑이 아닌)

```python
# 개념적 처리 과정
bank_embedding = initial_embedding("bank")

# Attention 후: 문맥 정보 추가
bank_with_context = attention_result  # deposits, money 정보 포함

# FFN 후: 의미 확정
bank_financial = ffn_transform(bank_with_context)  # 금융기관 의미로 변환
```

## 한계와 개선점

### 현재 FFN의 제약

**스터디에서 발견한 한계:**

> **지영**: "각 토큰이 독립적으로 처리되니까 토큰 간 상호작용이 제한적이네."
> 
> **현우**: "그리고 항상 같은 변환을 적용하니까 동적이지 못해."

### 최신 개선 방향

최근 연구들은 이런 한계를 극복하려고 해요:

- **Gated FFN**: 변환 강도를 동적으로 조절
- **Mixture of Experts (MoE)**: 여러 전문가 FFN 중 선택
- **GLU variants**: 더 효과적인 게이팅 메커니즘

## 다음 편 예고: 안정성의 기둥들

> *지금까지 정보 처리의 핵심(Attention, FFN)을 봤다면, 이제 이 모든 것을 안정적으로 작동시키는 기반 기술을 알아볼 차례입니다. Residual Connection과 Layer Normalization - 깊은 네트워크를 가능하게 하는 두 기둥의 비밀을 파헤쳐봅시다.*

**스터디에서 나온 다음 궁금증:**
"Add & Norm이 뭐길래 이렇게 중요하다고 하지? 없으면 안 되나?"

### 다음 편에서 다룰 내용
- 기울기 소실 문제와 해결책
- Batch Norm vs Layer Norm의 차이
- Pre-LN vs Post-LN 구조 비교
- 실험: 정규화 없이 학습하면?

## 마무리하며

FFN을 이해하고 나니 Transformer의 설계가 더 명확해졌어요. Attention이 "정보 수집"을 담당한다면, FFN은 "정보 가공"을 담당하는 거죠.

핵심을 정리하면:
- **Position-wise**: 각 토큰 독립적 처리
- **비선형성**: ReLU로 복잡한 패턴 학습 가능
- **차원 확장**: 표현력 증대 후 원래 차원으로 압축
- **Attention과의 협업**: 수집된 정보를 의미 있게 변환

다음 편에서는 이 모든 처리가 안정적으로 이뤄지도록 하는 Residual Connection과 Layer Normalization을 알아보겠습니다.

**FFN 없는 Transformer를 상상해보세요. 정보는 섞이지만 변환되지 않는, 반쪽짜리 모델이 될 거예요.**

---

*P.S. 가장 단순해 보이는 구성요소가 실제로는 가장 중요한 역할을 한다는 걸 다시 한번 깨달았어요. 단순함 속의 깊이랄까요.*

**다음 편에서 만나요!**