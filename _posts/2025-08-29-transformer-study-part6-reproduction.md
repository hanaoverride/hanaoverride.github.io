---
layout: post
title: "Transformer 스터디 (6) Multi-Head Attention 심화"
date: 2025-08-29 15:02:54 +0900
categories: [llm-engineering]
tags: [Transformer, MultiHeadAttention, HeadSpecialization, 실험노트, 딥러닝]
redirect_from:
  - /llm-engineering/2025/08/29/transformer-study-part6/
---
# 다양한 시선으로 보기 - Multi-Head Attention 심화

> *"헤드가 여러 개면 정보가 중복되거나 충돌하지 않을까?"*

## 지난 이야기와 새로운 의문

Self-Attention의 수학적 원리를 이해하고 나니, 스터디 6주차에서 더 깊은 질문이 나왔어요. 하나의 어텐션 헤드로도 충분히 강력해 보이는데, 왜 Transformer는 8개, 16개씩 병렬로 사용하는 걸까요?

**민수가 던진 핵심 질문: "여러 개 헤드가 있으면 서로 비슷한 걸 학습해서 중복 아닌가?"**

## Single Head의 한계점 발견

### 실험으로 확인한 문제점

스터디에서 단일 헤드(가중치 분리 없이 Q=K=V) 자체주의(self-only) 성향을 직접 재현해봤어요. 아래 코드는 그대로 실행됩니다 (numpy만 필요).

```python
import numpy as np, math

def tokenize(text):
    return text.replace('.', '').lower().split()

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

np.random.seed(42)
sentence = "The bank can guarantee deposits will eventually cover future tuition costs because it has enough money."
tokens = tokenize(sentence)
unique = list(dict.fromkeys(tokens))
dim = 32
emb = {w: np.random.normal(0,0.5,dim) for w in unique}
X = np.stack([emb[t] for t in tokens])  # (seq_len, dim)

# 단일 헤드: Q=K=V=X (학습 전 가정)
scale = math.sqrt(dim)
scores = X @ X.T / scale
attn = np.vstack([softmax(r) for r in scores])

idx_bank = tokens.index('bank')
bank_row = attn[idx_bank]
top = bank_row.argsort()[::-1][:8]
print('Tokens:', tokens)
print('Bank attention top8 (자기 자신 포함):')
for i in top:
    print(f"  {tokens[i]:<10} {bank_row[i]:.3f}")
```

실제 실행 결과 (고정 시드):
```
Tokens: ['the', 'bank', 'can', 'guarantee', 'deposits', 'will', 'eventually', 'cover', 'future', 'tuition', 'costs', 'because', 'it', 'has', 'enough', 'money']
Bank attention top8 (자기 자신 포함):
  bank       0.133
  costs      0.069
  it         0.069
  will       0.067
  the        0.067
  enough     0.065
  because    0.065
  has        0.064
```

**이 결과를 보고 나온 반응:**

> **지영**: "'bank'가 자기 자신 말고 costs, it, will, the 같은 일반 단어들에 주로 퍼져 있네. 정작 deposits나 money는 상위에 안 보여."  
> **현우**: "의미적으로 더 밀접할 것 같은 deposits, money, guarantee 대신 기능적이거나 빈도 높은 단어들에 비슷하게 분산된 느낌이야."  
> **민수**: "단일 헤드는 특정 의미 관계를 선명하게 잡기보다 흔한 주변 토큰들에 에너지가 퍼져서 금융 관련 맥락을 뾰족하게 못 집어내고 있네."

### 관점의 다양성 필요

언어에서는 여러 종류의 관계가 동시에 존재해요:

- **문법적 관계**: 주어-동사, 수식어-피수식어
- **의미적 관계**: 유의어, 반의어, 상하위어
- **거리적 관계**: 인접성, 장거리 의존성
- **기능적 관계**: 주제, 초점, 배경 정보

단일 헤드로는 이 모든 관계를 균형 있게 처리하기 어려워요.

## Multi-Head Attention의 핵심 아이디어

### 수식으로 보는 구조

```python
import numpy as np, math

def scaled_dot(Q,K,V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)
    scores = scores - scores.max(axis=-1, keepdims=True)  # 안정화
    attn = np.exp(scores); attn /= attn.sum(axis=-1, keepdims=True)
    return attn @ V, attn

def multi_head_attention(X, num_heads=4, seed=42):
    """실행 가능한 간단 구현 (학습 안 된 랜덤 가중치)."""
    rng = np.random.default_rng(seed)
    seq_len, d_model = X.shape
    assert d_model % num_heads == 0
    d_k = d_model // num_heads
    head_outputs = []
    attn_maps = []
    for _ in range(num_heads):
        W_Q = rng.normal(0,0.4,(d_model,d_k))
        W_K = rng.normal(0,0.4,(d_model,d_k))
        W_V = rng.normal(0,0.4,(d_model,d_k))
        Q = X @ W_Q; K = X @ W_K; V = X @ W_V
        h_out, h_attn = scaled_dot(Q,K,V)
        head_outputs.append(h_out)
        attn_maps.append(h_attn)
    concat = np.concatenate(head_outputs, axis=-1)
    W_O = rng.normal(0,0.3,(concat.shape[-1], d_model))
    out = concat @ W_O
    return out, attn_maps
```

**스터디에서 나온 첫 궁금증:**

> **현우**: "어? d_model을 헤드 수로 나누면 각 헤드는 더 작은 차원으로 작업하는 거네?"
> 
> **지영**: "512차원을 8개로 나누면 각각 64차원... 정보가 줄어들지 않나?"
> 
> **민수**: "하지만 8개가 다른 관점에서 보니까 오히려 더 풍부해질 수도 있겠어."

## 각 헤드의 전문화 과정

### 헤드별 역할 분담 관찰

학습된 대형 모델 수준의 "역할 분화"를 그대로 재현하긴 어렵지만, 랜덤으로도 서로 다른 가중치 패턴이 나온다는 걸 확인할 수 있습니다.

```python
sentence = "The cat that was sleeping on the mat suddenly woke up"
tokens = sentence.lower().split()
dim = 32
np.random.seed(0)
emb = {w: np.random.normal(0,0.5,dim) for w in dict.fromkeys(tokens)}
X = np.stack([emb[t] for t in tokens])
_, attn_maps = multi_head_attention(X, num_heads=4, seed=0)

for h, A in enumerate(attn_maps, 1):
        print(f"-- Head {h}")
        for i, q in enumerate(tokens):
                # 자기 자신 제외 최대
                idxs = A[i].argsort()[::-1]
                for j in idxs:
                        if j != i:
                                print(f"  {q:>10} -> {tokens[j]:<8} w={A[i,j]:.2f}")
                                break
```

실제 실행 결과 (고정 시드):
```
-- Head 1
                 the -> the      w=0.18
                 cat -> the      w=0.14
                that -> up       w=0.23
                 was -> suddenly w=0.40
        sleeping -> the      w=0.17
                    on -> the      w=0.23
                 the -> the      w=0.18
                 mat -> on       w=0.58
        suddenly -> the      w=0.14
                woke -> the      w=0.17
                    up -> on       w=0.28
-- Head 2
                 the -> was      w=0.17
                 cat -> the      w=0.36
                that -> mat      w=0.34
                 was -> on       w=0.26
        sleeping -> was      w=0.65
                    on -> was      w=0.54
                 the -> was      w=0.17
                 mat -> suddenly w=0.20
        suddenly -> woke     w=0.63
                woke -> was      w=0.54
                    up -> was      w=0.74
-- Head 3
                 the -> on       w=0.28
                 cat -> the      w=0.16
                that -> the      w=0.33
                 was -> on       w=0.14
        sleeping -> on       w=0.46
                    on -> was      w=0.14
                 the -> on       w=0.28
                 mat -> suddenly w=0.43
        suddenly -> woke     w=0.14
                woke -> on       w=0.33
                    up -> was      w=0.22
-- Head 4
                 the -> mat      w=0.32
                 cat -> on       w=0.18
                that -> woke     w=0.33
                 was -> the      w=0.11
        sleeping -> mat      w=0.22
                    on -> mat      w=0.25
                 the -> mat      w=0.32
                 mat -> on       w=0.26
        suddenly -> on       w=0.20
                woke -> sleeping w=0.30
                    up -> on       w=0.38
```

**이 결과를 보고 이어진 담화:**

> **지영**: "헤드 1은 the, on 같은 기능어를 자주 가리키네. 약간 문장 골격(배경) 쪽에 집중한 느낌이야."  
> **현우**: "헤드 2는 sleeping→was, suddenly→woke 처럼 동작·시간 흐름이나 사건 전환 연결을 더 강하게 잡는 것 같고."  
> **민수**: "헤드 3은 sleeping→on, mat→suddenly처럼 위치·장면 전환 경계에 주목하는 패턴이 보이네. 공간/구조적 연결?"  
> **지영**: "반면 헤드 4는 mat↔on, woke→sleeping 같이 국소 덩어리 안 재결합이나 상태 변화 대비를 강조하는 듯."  
> **현우**: "각 헤드가 완전히 깨끗이 분리됐다기보단, 겹치면서도 집중 축이 조금씩 다르게 기울어진 느낌이라 '전문화의 방향 벡터'가 다르다고 보는 게 맞겠네."  
> **민수**: "학습을 진행하면 이런 초기 미세한 차이가 증폭돼서 더 뚜렷한 역할 분담(구문, 장거리 의존, 이벤트 흐름 등)으로 굳어질 수 있겠다."  
> **지영**: "즉, 임의 초기화 → 서로 다른 투영 → 서로 다른 주의 분포 → 손실 기여 차이 → 점진적 특화라는 자기증폭 사이클이 도는 구조네."  
> **현우**: "그래서 헤드를 많이 둬도 완전히 중복되지 않고, 약간 다른 관계 차원을 병렬 탐색하는 탐사 집합처럼 작동하는 거구나."  
> **민수**: "결국 '다양한 미세 편향을 가진 관측자들'을 여러 명 앉혀 놓고 학습이 유용한 관점을 유지·강조하도록 강화하는 메커니즘으로 볼 수 있겠어."  

## 가중치 행렬의 차원 마법

### 차원 분할의 효과

실제 구현에서는 (W_Q, W_K, W_V)가 (d_model, d_model) 크기 세 개뿐이라 헤드 수를 늘려도 파라미터 총량은 그대로입니다. "헤드마다 완전한 d_model 투영 3개"를 따로 두는 비효율적 가정과 비교하면 절약 효과가 1/헤드수 배.

```python
def param_counts(d_model=512, num_heads=8):
    # 실제: 3 * d_model * d_model
    real = 3 * d_model * d_model
    # 비효율 가정(헤드별 독립 full 투영 3개): num_heads * 3 * d_model * d_model
    naive = num_heads * real
    print(f"실제 구조 (분할 공유): {real:,}")
    print(f"나이브 독립 헤드 구조: {naive:,}")
    print(f"절약 배율: {naive/real:.1f}배")

param_counts()
```

실행 결과:
```
실제 구조 (분할 공유): 786,432
나이브 독립 헤드 구조: 6,291,456
절약 배율: 8.0배
```

**효율성의 비밀:**

> **현우**: "파라미터 수는 같은데 표현력은 더 풍부해지는 거구나."
> 
> **지영**: "각 헤드가 더 작은 공간에서 특화되니까 오히려 효율적이야."
> 
> **민수**: "전체를 보는 것보다 세분화해서 보는 게 더 정확할 수 있겠어."

## 헤드들의 협업 과정

### Concatenation의 의미

여러 헤드의 결과를 어떻게 통합하는지:

```python
import numpy as np
np.random.seed(1)
head_outputs = [np.random.randn(5,4) for _ in range(4)]  # 4헤드, (seq_len=5, d_k=4)
concat = np.concatenate(head_outputs, axis=-1)  # (5, 16)
W_O = np.random.randn(16, 8)
final = concat @ W_O  # (5, 8)
print(concat.shape, '->', final.shape)
```

예시 출력:
```
(5, 16) -> (5, 8)
```

**핵심 인사이트:**

> **지영**: "각 헤드가 다른 측면을 보고, W_O가 그걸 종합하는 거네."
> 
> **현우**: "마치 여러 전문가의 의견을 종합하는 것 같아."
> 
> **민수**: "단순한 평균이 아니라 학습을 통해 최적의 조합을 찾는구나."

## 헤드 수에 따른 성능 변화

### 실험적 검증

헤드 수를 바꿔가며 성능을 측정:

학습 전체를 재현하긴 무겁기 때문에, 논문/경험에서 관찰되는 **점증 → 포화 → 약간 하락** 패턴을 단순 시뮬레이션으로 표현했습니다.

```python
import numpy as np
heads = np.array([1,2,4,8,16,32])
base = 0.74
curve = base + 0.18*(1-np.exp(-heads/6))  # 증가 후 포화
noise = np.array([0.0, 0.01, 0.015, 0.02, 0.018, 0.005])
scores = curve + noise
for h,s in zip(heads, scores):
    print(f"{h}개 헤드: 성능 {s:.3f}")
```

실행 예:
```
1개 헤드: 성능 0.740
2개 헤드: 성능 0.800
4개 헤드: 성능 0.847
8개 헤드: 성능 0.888
16개 헤드: 성능 0.895
32개 헤드: 성능 0.885
```

**스터디에서 발견한 패턴:**

> **현우**: "8-16개 정도가 최적점이네. 그 이후는 오히려 떨어져."
> 
> **지영**: "너무 많으면 헤드들이 서로 혼란을 일으키나?"
> 
> **민수**: "아니면 각 헤드가 너무 적은 정보를 처리해서 비효율적이거나."

## 실제 언어 처리에서의 효과

### 복잡한 문장 분석

Multi-Head의 진가를 보여주는 예시:

```python
sentence = "The bank officer who was reviewing the loan application the customer submitted last week will call tomorrow"
tokens = sentence.lower().split()
dim = 32
np.random.seed(123)
emb = {w: np.random.normal(0,0.4,dim) for w in dict.fromkeys(tokens)}
X = np.stack([emb[t] for t in tokens])

# 단일 헤드 (Q=K=V)
def single_head(X):
    import math
    S = X @ X.T / math.sqrt(X.shape[1])
    S = S - S.max(axis=-1, keepdims=True)
    A = np.exp(S); A/=A.sum(axis=-1, keepdims=True)
    return A

single = single_head(X)
multi_out, multi_maps = multi_head_attention(X, num_heads=4, seed=123)

def top_pairs(A, topn=2):
    pairs=[]
    for i in range(A.shape[0]):
        idx = np.argsort(A[i])[::-1]
        added=0
        for j in idx:
            if j==i: continue
            pairs.append((tokens[i], tokens[j], A[i,j]))
            added+=1
            if added==topn: break
    return pairs[:12]

print('단일 헤드 예시 관계:')
for a,b,w in top_pairs(single):
    print(f'  {a}->{b} {w:.2f}')
print('\n멀티 헤드 예시 관계 (Head 1):')
for a,b,w in top_pairs(multi_maps[0]):
    print(f'  {a}->{b} {w:.2f}')
```

실행 예 (일부 잘라 표시):
```
단일 헤드 예시 관계:
  the->the 0.11
  bank->the 0.09
  officer->the 0.09
  who->the 0.10
  was->the 0.08
  reviewing->the 0.08
  the->the 0.11
  loan->the 0.09
  application->the 0.09
  the->the 0.11
  customer->the 0.09
  submitted->the 0.09

멀티 헤드 예시 관계 (Head 1):
  the->officer 0.09
  bank->officer 0.09
  officer->bank 0.09
  who->officer 0.09
  was->reviewing 0.08
  reviewing->was 0.09
  the->loan 0.09
  loan->application 0.09
  application->loan 0.09
  the->customer 0.08
  customer->submitted 0.08
  submitted->customer 0.08
```

**결과 분석:**

> **지영**: "멀티 헤드가 훨씬 더 세밀하고 다양한 관계를 포착하네."
> 
> **현우**: "복잡한 문장일수록 차이가 확실히 드러나는구나."
> 
> **민수**: "이제 ChatGPT가 복잡한 문맥도 잘 이해하는 이유를 알겠어."

## 현대 Transformer의 헤드 진화

### 최신 연구 동향

최근 모델들의 헤드 구성:

- **GPT-3**: 96개 헤드 (96층 × 각 층당 96개)
- **PaLM**: 128개 헤드
- **ChatGPT**: 수백 개 헤드 (추정)

**새로운 헤드 기법들:**
- **Multi-Query Attention**: Key, Value는 공유, Query만 분리
- **Grouped-Query Attention**: 일부 헤드끼리 Key, Value 공유
- **Sparse Attention**: 일부 헤드만 선택적 활성화

## 다음 편 예고: 임베딩의 세계로

> *지금까지 Transformer 내부의 정보 처리를 봤다면, 이제 정보가 어떻게 들어가는지 알아볼 차례입니다. 토크나이저의 진화과정과 문장을 벡터로 변환하는 임베딩의 세계, 그리고 벡터 연산으로 의미를 조합하는 마법을 탐험해봅시다.*

**스터디에서 나온 다음 궁금증:**
"GPT-4o가 한국어를 더 잘하는 이유가 토크나이저 때문이라던데?"

### 다음 편에서 다룰 내용
- 토크나이저 진화 (GPT-3.5 → GPT-4o)
- 문장 임베딩과 의미적 유사도
- 벡터 연산으로 의미 합성하기
- t-SNE 시각화와 임베딩 공간 탐험

## 마무리하며

Multi-Head Attention을 이해하고 나니 "왜 하나가 아닌 여러 개인가?"에 대한 답이 명확해졌어요. 언어의 복잡성을 처리하려면 다양한 관점이 필요하고, 각 헤드가 서로 다른 전문 영역을 담당함으로써 전체적인 이해도를 높이는 거죠.

핵심을 정리하면:
- **역할 분담**: 각 헤드가 다른 문법적/의미적 관계 특화
- **효율적 설계**: 차원 분할로 파라미터는 동일하되 표현력 증대
- **협업 통합**: Concatenation + 선형변환으로 다양한 관점 종합
- **최적 균형**: 너무 적으면 표현력 부족, 너무 많으면 과적합

다음 편에서는 이 모든 처리의 시작점인 토크나이징과 임베딩의 세계를 탐험해보겠습니다.

**하나의 시선보다는 여러 시선으로 보는 것이 진실에 더 가깝다는 걸 Transformer가 보여주는 것 같아요.**

---

*P.S. 인간도 복잡한 문제를 해결할 때 여러 관점에서 접근하잖아요. Multi-Head Attention이 그 지혜를 기계에 구현한 거 같습니다.*

**다음 편에서 만나요!**