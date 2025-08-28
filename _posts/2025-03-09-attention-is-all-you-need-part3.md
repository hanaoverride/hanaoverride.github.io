---
layout: post
title: "Attention Is All You Need 논문 : 처음 읽어보기! - (3)"
date: 2025-03-09 21:53:45 +0900
categories: [llm-engineering]
tags: [Transformer, 논문리뷰, EncoderDecoder, ScaledDotProduct, 딥러닝]
redirect_from:
  - /llm-engineering/2025/03/09/attention-is-all-you-need-part3/
---
### **Part 3: "실전 인사이트" - 왜 이게 혁명인가?**

## 🤯 Why Self-Attention?: 진짜 장점이 뭔데?

자, 이제 핵심 질문입니다. **"그래서 Self-Attention이 왜 좋은데?"**

논문이 제시한 세 가지 이유를 실전 경험과 함께 보죠:

### 1️⃣ **계산 복잡도: O(n²) vs O(n)**

**RNN의 고통:**
```python
# RNN: 순차 처리 (병렬화 불가)
for t in range(sequence_length):
    hidden[t] = f(hidden[t-1], input[t])  # t-1 끝나야 t 시작
```

**Transformer의 자유:**
```python
# Self-Attention: 동시 처리 (완전 병렬화)
attention = compute_all_pairs(input)  # 모든 쌍 동시 계산!
```

**실제로 경험해보니:**
- 100단어 문장 처리 시간
  - RNN: 10초
  - Transformer: 0.1초
- "아니 이게 100배 차이가 난다고?" 네, 진짜입니다.

### 2️⃣ **장거리 의존성: 거리 상관없이 연결**

이 문장을 보세요:
> "그 사람이 **10년 전** 파리에서 샀던 **빵**은 정말 맛있었다"

- RNN: "10년 전"과 "빵" 사이 12단어... 연결 끊김 😵
- Transformer: "거리? 상관없어. 바로 연결!" 😎

**Maximum Path Length 비교:**
- RNN: O(n) - 멀수록 어려움
- Self-Attention: O(1) - 거리 무관!

### 3️⃣ **해석 가능성: 뭘 보고 있는지 알 수 있다**

Attention Weight를 시각화하면 **모델이 뭘 보는지 보입니다:**

```
"The cat sat on the mat"
       ↓ (attention)
    [cat] ← [sat] (주어-동사 관계)
    [mat] ← [on] (전치사-목적어 관계)
```

**"오, 진짜 문법 관계를 학습하네?"** 
맞습니다. 이게 바로 Interpretability의 힘입니다.

## 🚀 Training: 실전 팁과 함정들

논문의 Training 섹션에서 놓치기 쉬운 디테일들:

### Learning Rate Schedule: "Warmup이 핵심이다"

```python
# 논문의 learning rate 공식
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

**처음엔 이해 못했는데,** 실제로 해보니:
- Warmup 없이: 학습 폭발 💥
- Warmup 있이: 안정적 수렴 📈

**"아, 처음엔 조심조심, 나중엔 과감하게!"**

### Regularization: Dropout 0.1의 마법

> "We apply dropout to the output of each sub-layer"

**Dropout 0.1이 왜 중요한가?**
- 0.0: 과적합 지옥
- 0.2: 학습 너무 느림
- 0.1: 딱 좋은 균형점

> Dropout이 무엇인지 알고 싶다면, 제 깃허브 레포지토리의 데이터 과학자를 위한 쿡북을 참고하세요!
> [데이터 과학자를 위한 쿡북](https://github.com/hanaoverride/data-scientist-cookbook-for-korean)
## 💡 결과와 영향: 숫자로 보는 혁명

### BLEU Score 비교:
- 이전 SOTA: 27.3
- Transformer: **28.4** (EN-DE)

**"점수 1.1점 차이가 뭐가 대단해?"**

BLEU 1.1점 차이는:
- 이전: "Dog is animal good"
- Transformer: "A dog is a good animal"

완전 다른 수준이죠.

### 학습 비용:
- 이전 모델: $10,000+ (추정)
- Transformer: $500-1000

**"민주화된 AI"의 시작이었습니다.**

## 🔮 그 이후: Transformer가 바꾼 세상

2017년 이 논문 이후:
- 2018: BERT (구글)
- 2019: GPT-2 (OpenAI)
- 2020: GPT-3
- 2023: ChatGPT 폭발
- 2024: 모든 AI가 Transformer 기반

**"아, 그래서 다들 이 논문 읽으라고 하는구나!"**

## 🎬 마무리: 함께 읽어서 더 좋았던

3편에 걸쳐 "Attention Is All You Need"를 함께 읽어봤습니다.

**솔직히 처음 읽을 땐:**
- "수식 뭐 이리 많아?"
- "이게 왜 혁명이지?"
- "나만 모르는 건가?"

**지금은:**
- "아, Attention이 진짜 전부구나"
- "병렬 처리가 핵심이었어"
- "이래서 ChatGPT가 가능했구나"

혼자 읽었으면 포기했을 텐데, 이렇게 하나씩 뜯어보니 이해가 되네요.

**다음엔 BERT 논문도 함께 읽어볼까요?** 
댓글로 의견 남겨주세요! 🚀
