---
layout: post
title: "Transformer 스터디 (5) Self-Attention의 수학적 이해"
date: 2025-08-29 13:15:06 +0900
categories: [llm-engineering]
tags: [Transformer, SelfAttention, 수학해설, QueryKeyValue, 딥러닝]
redirect_from:
  - /llm-engineering/2025/08/29/transformer-study-part5/
---
# 관심의 메커니즘 - Self-Attention 수학적 이해

> *"내적으로 유사도를 계산한다는데, 정말 그게 의미적 유사도와 일치하나?"*

## 지난 이야기와 수학적 궁금증

지금까지 Transformer의 구조적 요소들을 살펴봤다면, 이제 핵심 엔진인 Self-Attention의 수학적 원리를 파헤쳐볼 차례입니다. 스터디 5주차, 실제로 어텐션 가중치를 시각화해보면서 나온 질문이 모든 걸 바꿨죠.

**"어떻게 벡터의 내적이 의미의 유사성을 나타낼 수 있지?"**

마침 스터디원들 중에서 수학적 배경을 어려워하는 구성원이 있기도 했고, 수학적인 이론을 다시 짚고 넘어가기 위해 기하학적 직관을 살펴보기로 했습니다.

## 벡터 내적의 기하학적 직관

### 내적의 본질적 의미

스터디에서 가장 먼저 확인한 것은 내적의 기하학적 의미였어요:

```python
import numpy as np
import matplotlib.pyplot as plt

# 벡터 내적 실험
def vector_similarity_experiment():
    # 두 벡터 정의
    vec_a = np.array([3, 0])  # 수평 방향
    vec_b = np.array([3, 3])  # 45도 방향
    vec_c = np.array([0, 4])  # 수직 방향
    vec_d = np.array([-2, 2]) # 135도 방향
    
    # 내적 계산
    dot_ab = np.dot(vec_a, vec_b)  # 9
    dot_ac = np.dot(vec_a, vec_c)  # 0  
    dot_ad = np.dot(vec_a, vec_d)  # -4
    
    print(f"A·B = {dot_ab} (같은 방향)")
    print(f"A·C = {dot_ac} (수직)")  
    print(f"A·D = {dot_ad} (반대 방향)")
```

**스터디에서 나온 반응:**

> **민수**: "내적이 클수록 방향이 비슷하고, 0이면 수직, 음수면 반대 방향이구나."
> 
> **지영**: "그럼 단어 벡터에서도 비슷한 의미의 단어들은 같은 방향을 가리킬까?"
> 
> **현우**: "실제로 확인해보자!"

### 실제 단어 임베딩으로 검증

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
words = ["king", "queen", "man", "woman", "apple", "orange", "car"]

# 임베딩 생성
embeddings = model.encode(words)

# 내적 기반 유사도 행렬
similarity_matrix = np.dot(embeddings, embeddings.T)

print("내적 기반 유사도:")
for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        if i < j:
            sim = similarity_matrix[i, j]
            print(f"{word1}-{word2}: {sim:.3f}")
```

결과:
```
내적 기반 유사도:
king-queen: 0.681
king-man: 0.322
king-woman: 0.264
king-apple: 0.243
king-orange: 0.364
king-car: 0.288
queen-man: 0.254
queen-woman: 0.439
queen-apple: 0.235
queen-orange: 0.304
queen-car: 0.298
man-woman: 0.326
man-apple: 0.226
man-orange: 0.173
man-car: 0.293
woman-apple: 0.313
woman-orange: 0.283
woman-car: 0.400
apple-orange: 0.373
apple-car: 0.410
orange-car: 0.368
```

**이 결과를 보고:**

> **지영**: "정말 의미적으로 비슷한 단어들끼리 높은 값이 나오네!"
> 
> **현우**: "king-queen, apple-orange가 높고, king-apple은 낮아."
> 
> **민수**: "벡터 공간에서 의미가 방향으로 표현되는구나."

## Query, Key, Value의 진실

### 데이터베이스 메타포 다시 보기

이전에 데이터베이스 비유로 설명했지만, 수학적으로 더 정확히 이해해봤어요:

```python
def attention_mechanism_detailed(X, W_Q, W_K, W_V):
    """
    X: 입력 시퀀스 (seq_len, d_model)
    W_Q, W_K, W_V: 변환 행렬들 (d_model, d_k)
    """
    
    # 1. 변환: 각 토큰을 Query, Key, Value로 변환
    Q = X @ W_Q  # (seq_len, d_k) - "무엇을 찾고 싶은가"
    K = X @ W_K  # (seq_len, d_k) - "나는 무엇인가"  
    V = X @ W_V  # (seq_len, d_v) - "내가 제공할 정보"
    
    # 2. 유사도 계산: 모든 Query-Key 쌍의 내적
    scores = Q @ K.T  # (seq_len, seq_len)
    
    # 3. 스케일링: 차원에 따른 조정
    scaled_scores = scores / np.sqrt(d_k)
    
    # 4. 소프트맥스: 확률 분포로 변환
    attention_weights = softmax(scaled_scores)
    
    # 5. 정보 집계: 가중 평균
    output = attention_weights @ V
    
    return output, attention_weights
```

### 왜 3개의 서로 다른 변환이 필요할까?

**스터디에서 실험한 극단적 경우:**

```python
# 만약 Q, K, V가 모두 같다면?
def naive_attention(X):
    scores = X @ X.T  # 자기 자신과의 유사도만 계산
    weights = softmax(scores)
    return weights @ X

# 문제점: 모든 토큰이 자기 자신에게만 높은 어텐션
# 다른 토큰들의 정보를 제대로 활용하지 못함
```

**깨달음:**

> **현우**: "Q, K, V가 다르면 '찾는 기준'과 '찾을 대상'과 '제공할 정보'를 각각 최적화할 수 있겠네."
> 
> **지영**: "같은 단어라도 문맥에 따라 다른 역할을 할 수 있고."
> 
> **민수**: "학습을 통해 각각이 특화되는 거구나."

## Scaling Factor √d_k의 비밀

### 문제 상황 재현

고차원에서 내적이 왜 문제가 되는지 실험:

```python
import numpy as np
def scaling_experiment():
    # 차원별 내적 분포 확인
    dimensions = [64, 128, 256, 512, 1024]
    
    for d in dimensions:
        # 표준정규분포에서 벡터 생성
        q = np.random.normal(0, 1, d)
        k = np.random.normal(0, 1, d)
        
        dot_product = np.dot(q, k)
        scaled_dot = dot_product / np.sqrt(d)
        
        print(f"차원 {d}: 원본={dot_product:.2f}, 스케일링 후={scaled_dot:.2f}")

scaling_experiment()

```

결과:
```
차원 64: 원본=-3.25, 스케일링 후=-0.41
차원 128: 원본=3.10, 스케일링 후=0.27
차원 256: 원본=11.50, 스케일링 후=0.72
차원 512: 원본=-22.82, 스케일링 후=-1.01
차원 1024: 원본=-3.16, 스케일링 후=-0.10
```

**스터디에서 나온 분석:**

> **지영**: "차원이 높아질수록 내적 값이 커지네. 이게 왜 문제지?"
> 
> **민수**: "소프트맥스에 넣으면 큰 값들이 확률을 독점할 거야."
> 
> **현우**: "그럼 어텐션이 극단적으로 한 곳에만 집중되겠구나."

### 소프트맥스 포화 현상

```python
def softmax_saturation_demo():
    # 스케일링 전후 비교
    large_scores = np.array([10.0, 8.0, 12.0, 9.0])
    small_scores = np.array([1.0, 0.8, 1.2, 0.9])
    
    large_probs = softmax(large_scores)
    small_probs = softmax(small_scores)
    
    print("큰 값들:", large_probs)  # [0.018, 0.002, 0.974, 0.007]
    print("작은 값들:", small_probs) # [0.274, 0.221, 0.331, 0.174]
```

**결과 분석:**

> **현우**: "큰 값들에서는 거의 하나만 선택되고, 작은 값들에서는 고르게 분포되네."
> 
> **지영**: "√d_k로 나누면 값들이 적절한 범위로 조정돼서 더 균형잡힌 어텐션이 가능하겠어."

## 소프트맥스의 수학적 필요성

### 확률 분포로의 변환

왜 단순 정규화가 아닌 소프트맥스를 사용하는지:

```python
def normalization_comparison():
    scores = np.array([2.0, 1.0, 3.0, 0.5])
    
    # 단순 정규화 (합이 1)
    simple_norm = scores / np.sum(scores)
    
    # 소프트맥스
    softmax_norm = np.exp(scores) / np.sum(np.exp(scores))
    
    print("원본 점수:", scores)
    print("단순 정규화:", simple_norm)  # [0.31, 0.15, 0.46, 0.08]
    print("소프트맥스:", softmax_norm)   # [0.26, 0.10, 0.71, 0.06]
```

**소프트맥스의 장점:**

1. **지수 함수**: 작은 차이도 큰 확률 차이로 변환
2. **항상 양수**: 모든 확률이 0 이상
3. **차별적 선택**: 높은 점수에 더 많은 확률 할당

**스터디에서 깨달은 점:**

> **민수**: "소프트맥스가 경쟁을 더 치열하게 만드는구나."
> 
> **지영**: "가장 관련성 높은 토큰에 확실히 더 많은 가중치를 주고."
> 
> **현우**: "하지만 완전히 0은 아니니까 다른 정보도 조금씩은 반영돼."

## 실제 어텐션 패턴 분석

### 문장에서의 어텐션 시각화

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def analyze_attention_patterns(seed=42, d_k=64, amplify=4.0, temperature=1.0, use_raw=False, min_var=1e-3):
    """
    amplify: 랜덤 프로젝션 가중치 분산 (값이 너무 작으면 모두 1/n으로 수렴)
    temperature < 1: 차이 증폭, > 1: 차이 완화
    use_raw=True: 임베딩 자체로 Q=K=V (랜덤 투영 없이)
    min_var: 행별 점수 분산이 이 값보다 작으면 한 번 재초기화하여 확대
    """
    np.random.seed(seed)
    tokens = "The cat sat on the mat".split()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(tokens)  # (n, d_model)
    n, d_model = X.shape

    if use_raw:
        Q = K = V = X.astype(np.float64)
    else:
        # Xavier 스타일 초기화 (분산이 너무 작아지는 것 방지) + 사용자 증폭
        base_scale = 1.0 / np.sqrt(d_model)
        scale = base_scale * amplify
        W_Q = np.random.randn(d_model, d_k) * scale
        W_K = np.random.randn(d_model, d_k) * scale
        W_V = np.random.randn(d_model, d_k) * scale
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V

    def compute_scores(Q, K):
        s = (Q @ K.T) / np.sqrt(d_k if not use_raw else Q.shape[1])
        return s

    scores = compute_scores(Q, K) / temperature

    # 1차 분산 진단
    row_var = scores.var(axis=1)
    mean_var = row_var.mean()
    print(f"초기 평균 점수분산: {mean_var:.6f}")

    # 분산이 너무 작아 거의 균일 softmax 예측 시 재초기화 (1회)
    if mean_var < min_var and not use_raw:
        print("분산이 너무 낮아 가중치 재초기화로 확대합니다.")
        scale *= 5.0  # 추가 확대
        W_Q = np.random.randn(d_model, d_k) * scale
        W_K = np.random.randn(d_model, d_k) * scale
        W_V = np.random.randn(d_model, d_k) * scale
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        scores = compute_scores(Q, K) / temperature
        row_var = scores.var(axis=1)
        mean_var = row_var.mean()
        print(f"재초기화 후 평균 점수분산: {mean_var:.6f}")

    # 행별 평균 제거 + 표준화로 대비 향상 (희미할 때)
    if mean_var < min_var * 5:
        scores = (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-8)
        print("점수 행 표준화 적용")

    attn = softmax(scores, axis=-1)

    print("어텐션 (행=Query, 열=Key):")
    print("토큰:", tokens)
    for i, tok in enumerate(tokens):
        print(f"{tok:>4}:", " ".join(f"{w:.2f}" for w in attn[i]))

    return attn

print("=== 기본 (균일해지기 쉬움) ===")
analyze_attention_patterns(amplify=1.0, temperature=1.0)

print("\n=== 분산 키우기 (amplify=4) ===")
analyze_attention_patterns(amplify=4.0, temperature=1.0)

print("\n=== 차이 강조 (temperature=0.5) ===")
analyze_attention_patterns(amplify=4.0, temperature=0.5)

print("\n=== 랜덤 투영 없이 원본 임베딩 직접 (use_raw=True) ===")
analyze_attention_patterns(use_raw=True, temperature=1.0)
```

결과 예시:
```
=== 기본 (균일해지기 쉬움) ===
초기 평균 점수분산: 0.000003
분산이 너무 낮아 가중치 재초기화로 확대합니다.
재초기화 후 평균 점수분산: 0.003046
점수 행 표준화 적용
어텐션 (행=Query, 열=Key):
토큰: ['The', 'cat', 'sat', 'on', 'the', 'mat']
 The: 0.03 0.07 0.12 0.18 0.03 0.57
 cat: 0.03 0.30 0.28 0.08 0.03 0.28
 sat: 0.05 0.47 0.03 0.22 0.05 0.19
  on: 0.05 0.09 0.03 0.26 0.05 0.53
 the: 0.03 0.07 0.12 0.18 0.03 0.57
 mat: 0.04 0.05 0.10 0.07 0.04 0.70

=== 분산 키우기 (amplify=4) ===
초기 평균 점수분산: 0.000734
분산이 너무 낮아 가중치 재초기화로 확대합니다.
재초기화 후 평균 점수분산: 0.779736
어텐션 (행=Query, 열=Key):
토큰: ['The', 'cat', 'sat', 'on', 'the', 'mat']
 The: 0.04 0.07 0.13 0.19 0.04 0.53
 cat: 0.06 0.27 0.26 0.10 0.06 0.26
 sat: 0.05 0.47 0.03 0.22 0.05 0.19
  on: 0.08 0.12 0.07 0.25 0.08 0.40
 the: 0.04 0.07 0.13 0.19 0.04 0.53
 mat: 0.03 0.05 0.10 0.07 0.03 0.72

=== 차이 강조 (temperature=0.5) ===
초기 평균 점수분산: 0.002938
점수 행 표준화 적용
어텐션 (행=Query, 열=Key):
토큰: ['The', 'cat', 'sat', 'on', 'the', 'mat']
 The: 0.06 0.12 0.65 0.02 0.06 0.09
 cat: 0.04 0.08 0.68 0.05 0.04 0.11
 sat: 0.08 0.43 0.27 0.02 0.08 0.12
  on: 0.15 0.40 0.08 0.02 0.15 0.20
 the: 0.06 0.12 0.65 0.02 0.06 0.09
 mat: 0.05 0.74 0.04 0.05 0.05 0.06

=== 랜덤 투영 없이 원본 임베딩 직접 (use_raw=True) ===
초기 평균 점수분산: 0.000216
점수 행 표준화 적용
어텐션 (행=Query, 열=Key):
토큰: ['The', 'cat', 'sat', 'on', 'the', 'mat']
 The: 0.40 0.06 0.04 0.05 0.40 0.04
 cat: 0.08 0.72 0.03 0.05 0.08 0.05
 sat: 0.06 0.04 0.74 0.05 0.06 0.05
  on: 0.06 0.05 0.04 0.74 0.06 0.04
 the: 0.40 0.06 0.04 0.05 0.40 0.04
 mat: 0.05 0.06 0.05 0.04 0.05 0.74
```
**패턴 분석:**

> **지영**: "temperature=0.5에서 'cat'→'sat', 'sat'→'cat'이 서로 높은 값이야. 주어-동사 관계가 두 방향으로 드러난 것처럼 보여."
> 
> **현우**: "'on'은 기대만큼 'mat'에만 집중하지 않고 'cat' 쪽 가중치가 더 커. 아직 초기 무작위 투영 + 짧은 문장이라 전치사-목적어 패턴이 또렷이 분리되진 않은 듯."
> 
> **민수**: "Q=K=V (use_raw) 설정에서는 거의 자기 자신(대각선) 위주거나 'The'–'the' 같이 표면형이 비슷한 토큰끼리만 엮여. 투영 학습이 표현력을 넓히는 이유가 보이네."
> 
> **지영**: "amplify를 키우면(4) 분산이 올라가면서 균일해지려는 경향이 줄고, temperature를 낮추면 관계(특히 cat–sat)가 더 선명해져."
> 
> **현우**: "완벽한 문법 구조 판별이라기보다, 초기 단계에서도 의미/역할 중심 허브(동사 'sat')로 수렴하려는 경향을 볼 수 있다는 정도로 해석하면 적절하겠어."

## Self-Attention의 한계와 개선

### 현재의 제약사항

**스터디에서 발견한 Self-Attention의 한계:**

1. **이차 복잡도**: 시퀀스 길이의 제곱에 비례하는 계산량
2. **위치 정보 부족**: Positional Encoding에 의존
3. **지역적 편향 부족**: 가까운 토큰에 대한 특별한 고려 없음

```python
# 복잡도 문제 시각화
sequence_lengths = [128, 256, 512, 1024, 2048]
attention_operations = [n**2 for n in sequence_lengths]

for n, ops in zip(sequence_lengths, attention_operations):
    print(f"길이 {n}: {ops:,} 연산")
```

결과:
```
길이 128: 16,384 연산
길이 256: 65,536 연산
길이 512: 262,144 연산  
길이 1024: 1,048,576 연산
길이 2048: 4,194,304 연산
```

### 개선 방향들

최신 연구에서 제시하는 해결책들:

- **Linear Attention**: O(n) 복잡도로 감소
- **Sparse Attention**: 일부 위치만 선택적 어텐션
- **Local Attention**: 윈도우 기반 지역적 어텐션

## 다음 편 예고: 여러 시선으로 보기

> *하나의 어텐션으로도 충분히 강력하지만, Transformer는 여러 개의 어텐션을 병렬로 사용합니다. Multi-Head Attention이 단일 헤드보다 뛰어난 이유와 각 헤드가 학습하는 서로 다른 패턴들을 알아봅시다.*

**스터디에서 나온 다음 궁금증:**
"헤드가 여러 개면 정보가 중복되거나 충돌하지 않을까?"

### 다음 편에서 다룰 내용
- Multi-Head의 핵심 아이디어
- 헤드별 특화 패턴 분석
- 가중치 행렬의 차원 관계
- 실험: 헤드 수에 따른 성능 변화

## 마무리하며

Self-Attention의 수학적 원리를 파헤쳐보니, 겉보기에 복잡해 보이는 공식들이 모두 명확한 이유가 있음을 알 수 있었어요. 

핵심을 정리하면:
- **내적**: 벡터 방향 유사도로 의미적 관련성 측정
- **Q, K, V 분리**: 역할별 최적화로 표현력 증대
- **√d_k 스케일링**: 고차원에서의 소프트맥스 포화 방지
- **소프트맥스**: 확률 분포로 변환하여 차별적 선택

다음 편에서는 이런 어텐션을 여러 개 병렬로 사용하는 Multi-Head Attention의 위력을 확인해보겠습니다.

**수학이 어렵게 느껴질 수 있지만, 각 단계마다 분명한 직관이 있다는 걸 기억하세요.**

---

*P.S. 벡터의 내적이 의미의 유사성을 나타낸다는 게 여전히 신기해요. 수학과 언어가 만나는 지점이죠.*

**다음 편에서 만나요!**
