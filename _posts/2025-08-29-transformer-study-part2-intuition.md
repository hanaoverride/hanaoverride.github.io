---
layout: post
title: "Transformer 스터디 (2) Positional Encoding - 거리를 파악하는 감각"
date: 2025-08-29 09:27:48 +0900
categories: [llm-engineering]
tags: [Transformer, PositionalEncoding, 위치인코딩, 직관, 딥러닝]
redirect_from:
  - /llm-engineering/2025/08/29/transformer-study-part2/
---
# 위치를 기억하는 마법 - Positional Encoding 완전분석

> *"병렬처리하면서 순서도 기억한다고? 어떻게?"*

## 지난 이야기

지난 편에서 Transformer의 핵심 아이디어를 알아봤죠. RNN의 순차처리 방식을 버리고, 모든 단어가 동시에 모든 단어를 보는 Self-Attention 메커니즘. 하지만 스터디 중에 현우가 던진 질문 하나가 모든 걸 바꿨어요.

**"잠깐, 그럼 '나는 학교에 간다'와 '학교에 나는 간다'를 어떻게 구분하지?"**

맞습니다. 순서를 버렸는데 순서가 중요한 언어를 어떻게 처리할까요?

## 순서의 딜레마

### 문제 상황 재현

스터디 2주차, 실제로 코드를 짜보면서 만난 첫 번째 벽이었어요.

```python
# Attention만으로 처리했을 때
sentence1 = ["나는", "학교에", "간다"]
sentence2 = ["학교에", "나는", "간다"]  
sentence3 = ["간다", "나는", "학교에"]

# 동일한 단어들로 구성되어 있어서
# Attention 결과가 똑같이 나온다!
```

**스터디 토론:**

> **지영**: "이상하다... 의미가 완전히 다른 문장인데 같은 결과가 나와?"
> 
> **민수**: "Self-Attention이 단어 간 관계만 보니까 순서는 모르는 거네."
> 
> **현우**: "그럼 RNN이 더 나은 거 아닌가? 적어도 순서는 알잖아."

바로 그 순간이었어요. Transformer의 가장 큰 약점을 발견한 거죠.

### 순서 정보의 중요성

언어에서 순서가 얼마나 중요한지 생각해보세요:

- "개가 고양이를 쫓는다" ≠ "고양이가 개를 쫓는다"
- "나는 어제 친구를 만났다" ≠ "어제 나는 친구를 만났다" (미묘한 뉘앙스 차이)
- "Not bad"와 "Bad not" (완전히 다른 의미)

순서를 잃으면 언어의 핵심을 잃는 거죠.

## Positional Encoding의 등장

### 해결책의 아이디어

Transformer 논문의 저자들은 천재적인 해결책을 제시했어요:

**"위치 정보를 벡터로 만들어서 단어 임베딩에 더해주자!"**

```python
# 핵심 아이디어
word_embedding = [0.5, 0.8, 0.2, ...]      # 단어의 의미
position_encoding = [0.0, 1.0, 0.0, ...]   # 위치 정보

transformer_input = word_embedding + position_encoding
```

**스터디에서 나온 첫 반응:**

> **민수**: "더하기? 그냥 더해도 되는 거야?"
> 
> **지영**: "정보가 섞이지 않을까?"
> 
> **현우**: "벡터 공간에서는 더하기가 정보를 합치는 거잖아. 그런데 정말 잘 될까?"

이 의심은 당연했어요. 하지만 실험해보니 놀라운 결과가...

## Sine/Cosine의 마법

### 왜 하필 삼각함수?

논문에서 제시한 Positional Encoding 공식을 보면 복잡해 보여요:

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**스터디에서 이 공식을 처음 봤을 때:**

> **지영**: "이게 뭐야... 왜 이렇게 복잡하지?"
> 
> **현우**: "그냥 [1, 2, 3, 4, ...] 이렇게 하면 안 되나?"
> 
> **민수**: "10000은 또 어디서 나온 거야?"

### 삼각함수를 선택한 이유

직접 실험해보면서 깨달았어요. 단순한 숫자 시퀀스(1, 2, 3, ...)의 문제점들:

1. **고정 길이 제한**: 학습할 때 본 길이보다 긴 문장이 오면?
2. **값의 폭발**: 위치 값이 너무 커져서 단어 임베딩을 압도
3. **상대적 거리 인식 어려움**: 위치 100과 101의 관계 vs 위치 1과 2의 관계

삼각함수는 이 모든 문제를 해결해요:

```python
import numpy as np

def get_positional_encoding(seq_len, d_model):
    """실제 구현 코드"""
    pos_encoding = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(d_model):
            angle = pos / (10000 ** (2 * i / d_model))
            
            if i % 2 == 0:  # 짝수 인덱스
                pos_encoding[pos, i] = np.sin(angle)
            else:  # 홀수 인덱스  
                pos_encoding[pos, i] = np.cos(angle)
                
    return pos_encoding
```

### 실험으로 확인하기

스터디에서 직접 실행해본 코드:

```python
# 5개 토큰, 6차원으로 테스트
seq_len, d_model = 5, 6
pos_encoding = get_positional_encoding(seq_len, d_model)

print("Positional Encoding:")
print(pos_encoding.round(3))
```

결과:
```
[[0.000  1.000  0.000  1.000  0.000  1.000]
 [0.841  0.999  0.002  1.000  0.000  1.000]
 [0.909  0.996  0.004  1.000  0.000  1.000]
 [0.141  0.990  0.006  1.000  0.000  1.000]
 [-0.757 0.983  0.009  1.000  0.000  1.000]]
```

**이 결과를 보고 나온 반응:**

> **현우**: "각 위치마다 완전히 다른 패턴이네!"
> 
> **지영**: "그리고 값들이 -1과 1 사이에 있어서 단어 임베딩과 비슷한 스케일이야."
> 
> **민수**: "10000이 이런 역할을 하는구나... 주파수를 조절하는 거네."

## 위치 정보가 어텐션에 미치는 영향

### 실제 효과 검증

가장 중요한 건 이 위치 정보가 정말 작동하는지 확인하는 거였어요. 스터디에서 실험한 코드:

```python
# 위치 인코딩 유무 비교 실험
import numpy as np

def scaled_dot_product_attention(query, key, value, d_k):
    scores = np.matmul(query, key.T) / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = np.matmul(attention_weights, value)
    return output, attention_weights

# 임의의 토큰 임베딩
token_embeddings = np.random.randn(5, 6)

# Case 1: Positional Encoding 포함
with_pe = token_embeddings + pos_encoding
attention_output_1, weights_1 = scaled_dot_product_attention(
    with_pe, with_pe, with_pe, 6
)

# Case 2: Positional Encoding 없음  
without_pe = token_embeddings
attention_output_2, weights_2 = scaled_dot_product_attention(
    without_pe, without_pe, without_pe, 6
)

print("어텐션 가중치 차이:", np.mean(np.abs(weights_1 - weights_2)))
print("어텐션 출력 차이:", np.mean(np.abs(attention_output_1 - attention_output_2)))
```

결과:
```
어텐션 가중치 차이: 0.0882
어텐션 출력 차이: 0.6805
```

**이 결과를 보고:**

> **지영**: "확실히 다르네! 위치 정보가 어텐션 패턴을 바꿔놓았어."
> 
> **현우**: "이제 같은 단어라도 위치에 따라 다른 의미로 처리되겠네."
> 
> **민수**: "더하기만 했는데 이렇게 큰 차이가 날 줄이야..."

## 상대적 위치 관계의 비밀

### 삼각함수의 숨겨진 능력

가장 놀라운 건 삼각함수의 특성 때문에 **상대적 위치 관계**도 학습할 수 있다는 거였어요.

```python
# 삼각함수의 덧셈 공식 활용
# sin(A + B) = sin(A)cos(B) + cos(A)sin(B)
# cos(A + B) = cos(A)cos(B) - sin(A)sin(B)

# 이는 PE(pos + k)가 PE(pos)와 PE(k)의 선형 결합으로 표현 가능함을 의미!
```

**스터디 마지막 주에 깨달은 점:**

> **현우**: "아! 그럼 모델이 '이 단어와 3칸 떨어진 단어' 같은 패턴을 학습할 수 있겠네?"
> 
> **지영**: "맞아! 절대 위치뿐만 아니라 상대 위치도 알 수 있다는 거야."

## 실무에서의 의미

### 왜 중요한가?

Positional Encoding이 없다면:

```python
# 이 모든 문장들이 같게 처리됨
sentences = [
    "나는 어제 친구를 만났다",
    "어제 나는 친구를 만났다", 
    "친구를 어제 나는 만났다",
    "만났다 친구를 어제 나는"
]
```

하지만 Positional Encoding이 있으면 각각 다른 의미로 올바르게 구분해요.

### ChatGPT의 문맥 이해

이제 ChatGPT가 긴 대화에서도 문맥을 잘 기억하는 이유를 알 수 있어요:

1. 각 토큰이 고유한 위치 정보를 가짐
2. Self-Attention이 위치 관계를 고려해서 가중치 계산
3. 결과적으로 "3문장 전에 말한 내용"과 "방금 말한 내용"을 구분

## 한계와 개선점

### 현실적 제약

물론 완벽하지는 않아요:

**스터디에서 발견한 한계점들:**

> **민수**: "그런데 아무리 길어도 고정된 최대 길이가 있을 거 아냐?"
> 
> **지영**: "맞아. 그리고 정말 먼 거리의 단어들은 여전히 관계 파악이 어려울 것 같아."

최신 연구들은 이런 한계를 극복하려고 해요:
- **Rotary Position Embedding (RoPE)**: 회전 행렬을 사용해 절대 위치를 인코딩
- **Learned Position Embedding**: 위치를 학습으로 결정
- **Relative Position Encoding**: 상대 위치에 더 집중

## 다음 편 예고: Feed Forward Network

> *Attention으로 관계를 파악했다면, 이제 그 정보를 어떻게 변환할까요? "정보 처리의 실제 엔진" FFN의 비밀을 파헤쳐봅시다.*

**스터디에서 나온 다음 궁금증:**
"Attention만으로는 뭔가 부족해 보이는데, FFN은 정확히 뭘 하는 거지?"

### 다음 편에서 다룰 내용
- Position-wise 처리의 의미
- ReLU 활성화 함수의 역할  
- FFN이 없으면 어떻게 될까?
- 실제 정보 변환 과정 추적

## 마무리하며

Positional Encoding을 이해하고 나니 Transformer의 설계 철학이 보이기 시작했어요. **병렬처리의 효율성을 포기하지 않으면서도 순서 정보를 보존하는 우아한 해결책**이죠.

수학이 어려워 보일 수 있지만, 핵심은 단순해요:
- 각 위치마다 고유한 "지문"을 만들어주기
- 그 지문을 단어 의미와 더해서 "위치가 포함된 의미" 생성
- 결과적으로 같은 단어도 위치에 따라 다르게 처리

다음 편에서는 이렇게 위치 정보가 담긴 임베딩을 실제로 어떻게 가공하는지, FFN의 역할을 자세히 알아보겠습니다.

**코드를 직접 실행해보시면서 위치 정보가 어떻게 어텐션 패턴을 바꾸는지 확인해보세요!**

---

*P.S. 삼각함수가 이렇게 언어처리에 쓰일 줄이야... 수학의 아름다움을 다시 한번 느꼈던 파트였어요.*

**다음 편에서 만나요!**