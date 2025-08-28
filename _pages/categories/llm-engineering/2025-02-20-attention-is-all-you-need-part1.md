---
layout: post
title: "Attention Is All You Need 논문 : 처음 읽어보기! - (1)"
date: 2025-02-20 18:20:24 +0900
categories: [llm-engineering]
tags: [Transformer, 논문리뷰, AttentionMechanism, 딥러닝]
---

### **Part 1: "왜 Transformer인가?" - RNN의 한계와 Attention의 등장**


## 🤔 이 논문, 왜 모두가 떠들썩한가?

"Attention Is All You Need"... 제목부터 뭔가 도발적이죠? 
2017년에 나온 이 논문 하나가 AI 판도를 완전히 바꿔놨습니다. ChatGPT, Claude, Gemini... 요즘 핫한 AI들 전부 이 논문에서 시작됐거든요. 현대 LLM 아키텍쳐는 대부분 이 Transformer 기반이거나 Transformer 변형 아키텍쳐를 사용합니다. LLM을 이해하고 싶다면 정말정말 Transformer만큼은 이해해야 합니다.

(Diffusion 등 다른 기술 기반 모델도 있지만, 오늘은 Transformer에 집중할게요.)

근데 솔직히 말하면, 논문 읽기 쉽지 않죠? 저도 처음 읽을 때 "아 이게 뭔 소리야..." 했습니다.
그래서 **함께 읽어보려고 합니다.** 혼자 읽으면 어려운데, 같이 읽으면 좀 덜 부담스럽잖아요?

> Disclaimer: 이 논문은 머신러닝과 딥러닝에 대한 기본 지식은 있어야 합니다. NLP 관련 지식은 없어도 읽을수 있지만, 딥러닝 관련 지식이 없다면 무슨 말인지 거의 이해할 수 없으므로 제 깃허브 레포지토리의 [데이터 과학자를 위한 쿡북](https://github.com/hanaoverride/data-scientist-cookbook-for-korean) 을 참고하세요.

## 📖 Introduction: "RNN아, 너 정말 최선이었니?"

논문이 시작하자마자 **RNN을 대놓고 까기 시작합니다.**

> "This inherently sequential nature **precludes** parallelization within training examples, which **becomes critical at longer sequence lengths**"

쉽게 말하면 이겁니다:
- RNN: "나는 단어를 하나씩 차례대로 처리해야 해... 😓"
- 우리: "아니 형, 지금 GPU 100개 놀고 있는데 병렬처리 좀 하자"
- RNN: "안돼... 나는 순서대로만 할 수 있어..."

**실제로 제가 겪었던 일인데요,** 저도 옛날 모델 공부 좀 해보려고 LSTM으로 번역 모델 돌려봤더니 문장 길어질수록 학습 시간이 기하급수적으로 늘어나더라고요. 100단어짜리 문장? 그냥 포기했습니다.

## 💡 그래서 뭐가 다른데?: Attention의 혁명

Transformer가 제안한 해법은 간단합니다:

> "The Transformer allows for significantly more parallelization and can reach a new **state of the art** in translation quality **after being trained for as little as twelve hours on eight P100 GPUs.**"

**12시간만에 SOTA 달성이라고요?** 
당시 다른 모델들은 며칠씩 돌려야 했거든요. 

비유하자면 이런 겁니다:
- **기존 RNN**: 책을 처음부터 끝까지 한 글자씩 읽는 사람
- **Transformer**: 책 전체를 한번에 펼쳐놓고 중요한 부분끼리 연결선 그으며 읽는 사람

어떤 게 더 효율적일까요? 당연히 후자죠.

## 🔍 Self-Attention: "나 자신을 돌아보다"

논문에서 핵심 개념인 Self-Attention을 이렇게 설명합니다:

> "**Self-attention**, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence."

음... 좀 추상적이죠? **실제 예시로 봅시다:**

```
문장: "은행에 가서 돈을 찾았다"
```

"은행"이라는 단어의 의미를 파악하려면?
- "돈"이라는 단어를 봐야 함 → 아, 금융기관이구나! (강둑 X)
- 모든 단어가 서로를 "주목(attention)"하면서 의미를 파악

**이게 바로 Self-Attention입니다.** 
RNN처럼 순서대로 읽지 않고, 모든 단어가 동시에 서로를 바라보는 거죠.

## 📊 성능은 정말 좋아졌나?

논문의 Table 2를 보면 충격적입니다:

![성능 비교표]({{ site.baseurl }}/assets/images/performance-table.png)

- **Training Cost**: 10²~10³배 차이 (base 모델 기준)
- **BLEU Score**: 당시 최고 기록 갱신

**"와, 이거 진짜네?"** 했던 기억이 나네요.
