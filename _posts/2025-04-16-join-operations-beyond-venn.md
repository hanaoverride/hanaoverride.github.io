---
layout: post
title: "JOIN 연산: 벤 다이어그램 모델은 그만"
date: 2025-04-16 11:59:22 +0900
categories: [computer-science]
tags: [데이터베이스, SQL, JOIN연산, 관계형모델]
redirect_from:
  - /computer-science/2025/04/16/join-operations-beyond-venn/
---
# JOIN 연산 배울 때 벤 다이어그램 그림... 이제 그만 보고 싶다

데이터베이스 공부하다 보면 **정말 지겹게 나오는 그림**이 하나 있죠.
네, 맞아요. 바로 그 동그라미 두 개 겹쳐놓고 "이게 JOIN입니다~" 하는 그거요.

![벤 다이어그램 예시]({{ site.baseurl }}/assets/images/img.png)

처음엔 "아~ 그렇구나" 했는데, 실무에서 JOIN 쓰다 보니까 뭔가 이상한 거예요.
**"어? 이거 완전 다른 개념 아닌가?"**

## 내가 느낀 찜찜함의 정체

솔직히 말할게요. 저는 이 설명 들을 때마다 속으로 이런 생각했거든요.

### 첫 번째 의문: "차원이 안 맞는데...?"
집합은 그냥 원소들의 모임. 그런데 테이블은 row × column의 **2차원 구조**.

### 두 번째 의문: "결과물이 완전 달라지는데?"
집합 합집합은 여전히 집합. 하지만 JOIN 결과는 컬럼/row 수 모두 달라짐.

## 그럼 뭐가 더 적절한 비유일까?

### 1. 데카르트 곱 (Cartesian Product) — 본질
```
A = {1, 2}
B = {a, b, c}
A × B = {(1,a), (1,b), (1,c), (2,a), (2,b), (2,c)}
```
== CROSS JOIN.

### 2. 확대 행렬 (Augmented)
키 기준 매칭 후 옆으로 붙이기 → LEFT/RIGHT JOIN 직관.

### 3. 행렬 곱 (Matrix Multiplication)
차원 변환 면에서 일부 유사하나 의미 압축 vs 데이터 결합 차이.

## 왜 이게 중요한가?

집합 연산은 사실 SQL에서 UNION / INTERSECT / EXCEPT.
벤 다이어그램 기반 학습은 혼란 유발.

## 결론
초보에게 직관은 줄 수 있지만 **근본 모델로는 부정확**.
JOIN = (Cartesian Product) + (조건 필터) + (투영/선택적 NULL 보강).

📚 더 읽어보기: [Say NO to Venn Diagrams When Explaining JOINs](https://blog.jooq.org/say-no-to-venn-diagrams-when-explaining-joins/)

*여러분은 JOIN 연산 처음 배울 때 어떤 설명이 가장 도움됐나요?*
