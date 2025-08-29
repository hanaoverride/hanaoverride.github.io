---
layout: post
title: "Transformer 스터디 (4) Residual Connection & Layer Normalization"
date: 2025-08-29 11:40:27 +0900
categories: [llm-engineering]
tags: [Transformer, ResidualConnection, LayerNorm, 안정성, 딥러닝]
redirect_from:
  - /llm-engineering/2025/08/29/transformer-study-part4/
---
# 안정성의 기반 - Residual Connection & Layer Normalization

> *"Add & Norm이 뭐길래 이렇게 중요하다고 하지? 없으면 안 되나?"*

## 지난 이야기와 새로운 발견

지금까지 Transformer의 핵심 처리 과정을 알아봤어요. Self-Attention으로 관계를 파악하고, FFN으로 정보를 변환하는 방식. 하지만 스터디 4주차에 실제 코드를 구현해보면서 예상치 못한 문제에 부딪혔어요.

**"어? 레이어를 몇 개만 쌓아도 학습이 안 되네?"**

## 깊은 네트워크의 저주

### 기울기 소실의 현실

스터디에서 직접 경험한 문제상황:

```python
# 순진한 Transformer 구현 (Residual Connection 없이)
class NaiveTransformerBlock(nn.Module):
    def __init__(self, d_model):
        self.attention = MultiHeadAttention(d_model)
        self.ffn = FeedForwardNetwork(d_model)
        
    def forward(self, x):
        # 그냥 순차적으로 처리
        x = self.attention(x)  # 문제의 시작
        x = self.ffn(x)
        return x

# 여러 블록을 쌓으면...
model = nn.Sequential([
    NaiveTransformerBlock(512) for _ in range(6)
])

# 결과: 학습이 거의 안 됨!
```

**스터디에서 나온 반응:**

> **민수**: "이상하다, 2-3층까지는 괜찮은데 6층부터 학습이 안 돼."
> 
> **지영**: "기울기를 확인해보니까 앞쪽 레이어는 거의 0에 가깝더라."
> 
> **현우**: "CNN에서도 비슷한 문제 봤는데... 기울기 소실이야?"

정확한 진단이었어요.

### 문제의 본질

깊은 네트워크에서 기울기가 역전파되면서 점점 작아지는 현상:

```python
# 기울기 역전파 과정 (개념적)
loss_gradient = 1.0

# 6층부터 거꾸로 전파
layer6_grad = loss_gradient * layer6_weight  # 0.8
layer5_grad = layer6_grad * layer5_weight    # 0.64  
layer4_grad = layer5_grad * layer4_weight    # 0.512
layer3_grad = layer4_grad * layer3_weight    # 0.41
layer2_grad = layer3_grad * layer2_weight    # 0.32
layer1_grad = layer2_grad * layer1_weight    # 0.26

# 앞쪽 레이어는 거의 업데이트 안 됨!
```

**이때 나온 깨달음:**

> **지영**: "곱셈이 계속 이어지니까 1보다 작은 값들이 계속 곱해지면서 0에 수렴하는구나."
> 
> **현우**: "그럼 앞쪽 레이어들은 학습이 거의 안 되겠네."
> 
> **민수**: "Transformer가 깊어질수록 성능이 좋아진다는데, 어떻게 이 문제를 해결한 거지?"

## Residual Connection의 등장

### 아이디어의 핵심

ResNet에서 처음 제안된 아이디어를 Transformer에 적용:

**"변환된 정보와 원본 정보를 더해주자!"**

```python

def residual_connection(x, sublayer):
    return x + sublayer(x)  # 핵심은 이 "+"

# Transformer 블록에서의 적용
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Attention with residual
        x = x + self.attention(x)
        
        # FFN with residual  
        x = x + self.ffn(x)
        
        return x
```

### 왜 이게 작동하는가?

**스터디에서 실험해본 기울기 흐름:**

```python

# Residual Connection이 있을 때
def backward_with_residual():
    # y = x + f(x)이므로
    # dy/dx = 1 + df/dx
    
    # 역전파 시:
    gradient_x = 1 + gradient_f  # 항상 1이 보장됨!
    
    # 곱셈이 아닌 덧셈으로 기울기 전파
    layer1_grad = base_grad + accumulated_changes
```

**이 실험 결과를 보고:**

> **현우**: "아! 1이 항상 더해지니까 기울기가 0으로 가지 않겠네."
> 
> **지영**: "원본 정보가 직접 전달되는 고속도로 같은 거구나."
> 
> **민수**: "그럼 깊게 쌓아도 앞쪽까지 정보가 전달되겠어."

### 정보 보존의 효과

Residual Connection의 또 다른 장점:

```python

# 정보 흐름 관점
input_info = original_embeddings + positional_encoding

# 여러 레이어를 거쳐도
layer1_output = input_info + attention1_changes + ffn1_changes  
layer2_output = input_info + accumulated_changes_1_to_2
layer6_output = input_info + accumulated_changes_1_to_6

# 원본 정보가 끝까지 보존됨!
```

## Layer Normalization의 필요성

### 학습 불안정성 문제

Residual Connection만으로도 많이 개선됐지만, 또 다른 문제가 있었어요:

```python
# 실제 스터디에서 관찰한 현상
epoch_1_activations = [0.1, 0.5, 0.8, 1.2, ...]
epoch_5_activations = [0.05, 2.1, 5.7, 0.3, ...]  
epoch_10_activations = [10.5, 0.01, 8.9, 15.2, ...]

# 값들이 들쭉날쭉, 학습이 불안정
```

**스터디 토론:**

> **민수**: "값의 분포가 계속 바뀌네. 이러면 다음 레이어가 적응하기 어려울 텐데."
> 
> **지영**: "Batch Normalization 같은 걸 써야 하나?"
> 
> **현우**: "근데 자연어는 배치 크기도 다르고 시퀀스 길이도 다른데..."

### Batch Norm vs Layer Norm

두 정규화 방식의 차이를 실험으로 확인:

```python

# Batch Normalization
batch = [
    [sent1_token1, sent1_token2, sent1_token3],
    [sent2_token1, sent2_token2, sent2_token3], 
    [sent3_token1, sent3_token2, sent3_token3]
]
# 같은 위치의 토큰들끼리 정규화 (세로 방향)

# Layer Normalization  
for sentence in batch:
    # 각 문장 내에서 정규화 (가로 방향)
    normalized_sent = normalize(sentence)
```

**왜 Layer Norm을 선택했을까?**

**스터디에서 발견한 Layer Norm의 장점:**

1. **배치 크기 독립적**: 배치 크기에 상관없이 동일하게 작동
2. **시퀀스 길이 무관**: 패딩 토큰의 영향을 받지 않음
3. **추론 시 일관성**: 학습과 추론 시 동일한 동작

```python

def layer_normalization(x, eps=1e-6):
    # 각 샘플의 모든 특성에 대해 정규화
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)

# 예시
input_vector = [1.0, 5.0, 3.0, 7.0]
normalized = layer_norm(input_vector)  # [-1.34, 0.45, -0.45, 1.34]
```

## Pre-LN vs Post-LN 논쟁

### 원본 논문의 방식 (Post-LN)

```python

def original_transformer_block(x):
    # Post-LN: 정규화가 마지막에
    x = layer_norm(x + self_attention(x))
    x = layer_norm(x + feed_forward(x))
    return x
```

### 개선된 방식 (Pre-LN)

```python

def improved_transformer_block(x):
    # Pre-LN: 정규화가 먼저
    x = x + self_attention(layer_norm(x))
    x = x + feed_forward(layer_norm(x))
    return x
```

**스터디에서 두 방식을 비교 실험:**

> **지영**: "Pre-LN이 학습이 더 안정적이네?"
> 
> **현우**: "학습률도 더 크게 설정할 수 있고."
> 
> **민수**: "GPT 같은 최신 모델들이 Pre-LN을 쓰는 이유가 있었구나."

### 안정성 차이의 이유

```python

# Post-LN의 문제점
residual_output = x + very_large_attention_output  
# 정규화 전에 값이 폭발할 수 있음

# Pre-LN의 장점
normalized_input = layer_norm(x)  # 항상 안정된 입력
stable_output = x + attention(normalized_input)  # 안정된 업데이트
```

## 실제 구현과 효과 검증

### 완전한 Add & Norm 블록

```python

class AddNormBlock(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, sublayer_fn):
        # Pre-LN 방식
        normalized = self.layer_norm(x)
        sublayer_output = sublayer_fn(normalized)
        dropped = self.dropout(sublayer_output)
        return x + dropped  # Residual connection

# 사용 예시
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Attention + Add & Norm
        x = self.add_norm1(x, lambda x: self.attention(x))
        
        # FFN + Add & Norm
        x = self.add_norm2(x, lambda x: self.ffn(x))
        
        return x
```

### 효과 검증 실험

스터디에서 진행한 비교 실험:

```python
# 실험 설정
configurations = [
    "baseline": no_residual_no_norm,
    "residual_only": with_residual, 
    "norm_only": with_layer_norm,
    "full_add_norm": with_both
]

# 6층 네트워크로 테스트
results = {
    "baseline": {"loss": "발산", "gradient": "소실"},
    "residual_only": {"loss": "불안정", "gradient": "전달됨"},
    "norm_only": {"loss": "느린수렴", "gradient": "소실"},  
    "full_add_norm": {"loss": "안정수렴", "gradient": "안정전달"}
}
```

**결과를 보고 나온 반응:**

> **현우**: "정말 둘 다 필요하구나. 하나만으론 부족해."
> 
> **지영**: "Residual은 정보 전달, LayerNorm은 학습 안정화."
> 
> **민수**: "이제 깊은 Transformer가 왜 가능한지 이해됐어."

## 현대 Transformer의 진화

### 최신 정규화 기법들

Add & Norm의 발전된 형태들:

```python
# RMSNorm (GPT-3부터 사용)
def rms_norm(x, eps=1e-6):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
    return x / (rms + eps)  # 평균 계산 생략으로 더 효율적

# DeepNorm (더 깊은 네트워크용) 
def deep_norm(x, alpha=0.87):
    return layer_norm(alpha * x + sublayer(x))  # 스케일링 추가
```

### Transformer 규모의 확장

Add & Norm 덕분에 가능해진 것들:

- GPT-3: 96개 레이어
- PaLM: 118개 레이어  
- ChatGPT: 수백 개 레이어 (추정)

**스터디 마지막 세션에서:**

> **지영**: "이 간단한 기법들로 이렇게 깊은 모델이 가능하다니."
> 
> **현우**: "Add & Norm 없었다면 GPT도 없었을 거야."
> 
> **민수**: "기초가 탄탄해야 높이 쌓을 수 있다는 걸 다시 깨달았어."

## 다음 편 예고: Self-Attention의 수학적 진실

> *이제 안정적인 기반 위에서 Transformer의 핵심인 Self-Attention의 수학적 원리를 파헤쳐볼 시간입니다. Query, Key, Value가 정확히 뭘 의미하는지, 내적과 소프트맥스가 왜 필요한지 수학적으로 완전히 이해해봅시다.*

**스터디에서 나온 다음 궁금증:**
"내적으로 유사도를 계산한다는데, 정말 그게 의미적 유사도와 일치하나?"

### 다음 편에서 다룰 내용
- 벡터 내적의 기하학적 의미
- 소프트맥스의 수학적 필요성  
- Scaling Factor √d_k의 비밀
- 실험: 문장 임베딩과 유사도 관계

## 마무리하며

Add & Norm을 이해하고 나니 Transformer가 왜 "혁명"이었는지 더 명확해졌어요. 단순히 Attention이라는 새로운 메커니즘 때문이 아니라, **깊은 네트워크를 안정적으로 학습시킬 수 있는 기반**을 만들었기 때문이죠.

핵심을 정리하면:
- **Residual Connection**: 기울기 소실 방지, 정보 보존
- **Layer Normalization**: 학습 안정화, 분포 정규화
- **Pre-LN**: 더 안정적인 학습을 위한 순서 최적화
- **협업 효과**: 둘이 함께해야 진정한 위력 발휘

다음 편에서는 이 안정적인 기반 위에서 작동하는 Self-Attention의 수학적 원리를 깊이 파보겠습니다.

**가장 화려한 기능(Attention)도 탄탄한 기반(Add & Norm) 없이는 무용지물이라는 걸 배웠어요.**

---

*P.S. 때로는 보이지 않는 기반 기술이 가장 중요하다는 걸 다시 한번 깨달았습니다. 개발도 마찬가지겠죠.*

**다음 편에서 만나요!**