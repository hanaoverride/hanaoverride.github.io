---
layout: post
title: "Attention Is All You Need 논문 : 처음 읽어보기! - (2)"
date: 2025-03-08 19:38:18 +0900
categories: [llm-engineering]
tags: [Transformer, 논문리뷰, RNN한계, AttentionMechanism, 딥러닝]
redirect_from:
  - /llm-engineering/2025/03/08/attention-is-all-you-need-part2/
---
### **Part 2: "Transformer 뜯어보기" - 핵심 구조 이해하기**

## 🏗️ 전체 구조: 반복되는 블록
Figure 1: 인코더 6, 디코더 6 (같은 구조 반복) → 깊이로 표현력 확보.

## 🎯 Scaled Dot-Product Attention
수식:
```
Attention(Q,K,V) = softmax(QK^T / √d_k) V
```
핵심: 유사도(Q·K)를 확률로 → Value 가중 합.

## 🐙 Multi-Head Attention
여러 "관점" 병렬 학습 → 문법/의미/위치 등 다른 서브스페이스.
Concat 후 선형변환으로 통합.

## 📍 Positional Encoding
순서 정보 부여 (sin, cos 주기적 패턴) → 절대+상대 위치 모두 간접 학습.

## 🔄 잔차 연결 & LayerNorm
Residual: 기울기 흐름 개선.
LayerNorm: 배치 크기와 무관하게 안정화.

## 🔌 Feed-Forward (Position-wise)
두 번의 Linear + ReLU (논문은 ReLU, 구현은 종종 GELU).
차원 확장(d_model→4*d_model) 후 축소.

## 🧱 Encoder layer 순서
1. Multi-Head Self-Attention (+ 잔차 + LayerNorm)
2. Position-wise FFN (+ 잔차 + LayerNorm)

## 📤 Decoder 추가 요소
1. Masked Self-Attention (미래 토큰 가림)
2. Encoder-Decoder Attention (소스 문장과 alignment)
3. FFN

## 🎛️ Mask 종류
- Padding mask: 패딩 토큰 영향 차단
- Subsequent(mask): 미래 정보 누출 방지

## 🗜️ 왜 Scaling 필요?
`QK^T` 값 분산 커짐 → softmax 큰 값 saturate → gradient vanishing 위험 → `1/√d_k` 로 안정화.

## ✅ 정리
- Attention = 병렬, 장기 의존 용이
- Multi-Head = 다양한 의미 투영
- PE = 순서 정보 주입
- Residual+Norm = 학습 안정성

다음 편(3): 학습 세부(optimizer, label smoothing, regularization) & 결과 지표.
