---
layout: post
title: "Attention Is All You Need 논문 : 처음 읽어보기! - (2)"
date: 2025-03-08 19:38:18 +0900
categories: [llm-engineering]
tags: [Transformer, 논문리뷰, RNN한계, AttentionMechanism, 딥러닝]
---

### **Part 2: "Transformer 뜯어보기" - 핵심 구조 이해하기**

## 🏗️ Transformer 전체 구조: 레고 블록처럼 생각하기

지난 글에서 "왜 Transformer인가?"를 봤으니, 이제 **"어떻게 만들어졌는가?"** 를 볼 차례입니다.

논문의 Figure 1이 바로 그 유명한 Transformer 구조도인데요:

> "The Transformer follows this overall architecture using **stacked self-attention and point-wise, fully connected layers** for both the encoder and decoder" [3]

복잡해 보이지만, **레고 블록 쌓기라고 생각하면 쉽습니다:**
- 인코더 블록 6개
- 디코더 블록 6개
- 각 블록은 동일한 구조 반복

**"아, 그냥 똑같은 거 6번 반복하는구나!"** 맞습니다. 진짜 그겁니다.

## 🎯 Scaled Dot-Product Attention: 핵심 중의 핵심

이제 진짜 중요한 부분입니다. Transformer의 심장이죠.

> "We call our particular attention "**Scaled Dot-Product Attention**"" [3]

수식 보면 겁나죠? 하지만 **SQL 아시는 분들은 이해가 빠를 겁니다:**

```sql
-- SQL의 JOIN과 비슷한 개념
SELECT value 
FROM table
WHERE key MATCHES query
ORDER BY similarity DESC
```

Attention도 마찬가지입니다:
1. **Query**: "나는 이런 정보를 찾고 있어"
2. **Key**: "나는 이런 정보를 갖고 있어"
3. **Value**: "자, 여기 그 정보야"

**실제 수식을 쉽게 풀어보면:**
```
Attention(Q,K,V) = softmax(QK^T/√dk)V
```

- `QK^T`: Query와 Key 얼마나 비슷한지 계산
- `/√dk`: 값이 너무 커지지 않게 조절 (안하면 gradient 터짐)
- `softmax`: 확률로 변환 (합이 1이 되게)
- `×V`: 확률에 따라 Value 가중 평균

**"아, 결국 비슷한 것끼리 연결하는 거구나!"** 네, 정확합니다.

## 🐙 Multi-Head Attention: 문어처럼 여러 관점으로 보기

Single Attention으론 부족했나봅니다:

> "Multi-head attention allows the model to **jointly** attend to information from different **representation subspaces**" [4]

**현실적인 비유로 설명하면:**

하나의 문장을 분석할 때:
- **Head 1**: "문법적 관계를 봐야지" (주어-동사 일치)
- **Head 2**: "의미적 관계를 봐야지" (단어 의미 연결)
- **Head 3**: "문맥을 봐야지" (전체 흐름)
- ...
- **Head 8**: "다른 관점도 봐야지"

이렇게 **8개의 다른 시각으로 동시에 분석**하는 겁니다.
문어가 8개 다리로 동시에 여러 일 하는 것처럼요.

## 📍 Positional Encoding: "순서를 잊지 말자"

Transformer의 약점이 하나 있었습니다:

**"어? 근데 단어 순서는 어떻게 알아?"**

RNN은 순서대로 처리하니 자연스럽게 순서를 알았는데, Transformer는 한번에 다 보니까 순서를 모르거든요.

그래서 나온 해법:
```python
# 위치 정보를 수학적으로 인코딩
PE(pos, 2i) = sin(pos/10000^(2i/dmodel))
PE(pos, 2i+1) = cos(pos/10000^(2i/dmodel))
```

**"뭔 삼각함수를 쓰고 난리야?"** 싶으시죠?

간단합니다. **각 위치마다 고유한 지문(fingerprint)을 만드는 겁니다.**
- sin, cos 조합으로 각 위치가 고유한 패턴을 가짐
- 상대적 위치 관계도 학습 가능

실제로 이걸 시각화하면 아름다운 패턴이 나옵니다. 마치 바코드처럼요.

## 💭 Feed-Forward Networks: "생각을 정리하는 시간"

각 Attention 레이어 다음엔 항상 FFN이 옵니다:

> "In addition to attention sub-layers, each of the layers contains a fully connected feed-forward network" [4]

**왜 필요할까요?**

Attention이 "정보 수집"이라면, FFN은 "정보 처리"입니다.
- Attention: "자, 관련 정보 다 모았어"
- FFN: "이제 이 정보들을 소화해서 의미를 추출하자"

두 단계 변환:
1. 차원 확장 (512 → 2048): 더 넓은 공간에서 생각
2. 차원 축소 (2048 → 512): 핵심만 압축

**마치 우리가 복잡한 문제를 풀 때처럼요.** 
정보 수집 → 머릿속에서 정리 → 핵심 추출.