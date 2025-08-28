---
layout: post
title: "JoinVisualizer: Join 연산의 카티션 곱 시각화 프로그램"
date: 2025-04-20 12:29:15 +0900
categories: [projects]
tags: [데이터베이스, JOIN연산, 시각화, Python, 교육도구]
redirect_from:
  - /projects/2025/04/20/join-visualizer/
---
지난번에 쓴 글([JOIN 연산: 벤 다이어그램 모델은 그만]({% post_url 2025-04-16-join-operations-beyond-venn %}))에서도 털어놨지만...
**벤 다이어그램으로 JOIN 설명하는 거, 진짜 별로예요**.

그래서 생각했죠.
**"차라리 내가 제대로 된 시각화 도구를 만들어버리자!"**

최근에 고민 끝에 내린 결론은? JOIN 연산과 가장 비슷한 수학 개념은 **tensor product**. (설명은 더 어려워지니 시각화로 대체)

## 이 프로그램 핵심: Cartesian Product

모든 JOIN은 사실 Cartesian Product + 필터링.
- INNER JOIN: 전체 조합 → 조건 일치만
- LEFT JOIN: 전체 조합 → 왼쪽 기준 보존 + NULL 채우기
- RIGHT JOIN: 오른쪽 기준
- FULL OUTER: 양쪽 모두 보존

## 기술 스펙
- 1920×1080 레이아웃
- OS 의존 라이브러리 미사용
- Python만 설치되어 있으면 실행 가능

## 배울 수 있는 것
1. JOIN = 곱 후 필터 개념 체화
2. JOIN 타입별 차이 한눈에
3. 추상 → 구체 전환 경험

## 개발 후기
FULL OUTER JOIN NULL 생성 과정을 시각화하며 개인적으로 제일 헷갈렸던 부분 정리.

## 참여
Issue 환영:
- 새로운 JOIN 모드
- UX 개선
- 버그 제보

**여러분은 JOIN 연산 처음 배울 때 뭐가 제일 어려웠나요?**
경험 공유 부탁드립니다!
