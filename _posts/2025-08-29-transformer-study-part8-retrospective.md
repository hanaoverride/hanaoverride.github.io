---
layout: post
title: "Transformer 스터디 (8) 트랜스포머 PoC 구현"
date: 2025-08-29 21:10:37 +0900
categories: [llm-engineering]
tags: [Transformer, 구현, PoC, 회고, 딥러닝]
redirect_from:
  - /llm-engineering/2025/08/29/transformer-study-part8/
---
# 모든 것을 하나로 - 실제 PoC로 돌아본 구현 회고

> *"이제 논문 속 블록 다 이해했으니, 진짜 우리가 만든 코드 기준으로 정리하자."*

## 무엇을 '직접' 만들었나

이번 편은 실제 제가 GitHub에 올린 PoC 레포지토리 `transformer-pytorch-poc` (모듈 분리형 Encoder 중심) 를 기준으로 회고합니다.

레포지토리는 다음 주소에서 찾아볼 수 있어요: [transformer-pytorch-poc](https://github.com/hanaoverride/transformer-pytorch-poc)

| 모듈 | 파일 | 핵심 책임 | 당시 의사결정 메모 |
|------|------|-----------|--------------------|
| 임베딩 | `embedding.py` | TokenEmbedding + PositionalEncoding | Torch 내장 `nn.Embedding` 그대로, position 은 precompute 후 slice |
| 어텐션 | `attention.py` | ScaledDotProductAttention, MultiHeadAttention | 초기에 einsum 고려 → 가독성 위해 matmul 유지 |
| FFN | `feed_forward.py` | 2-layer MLP (ReLU) | GELU vs ReLU 고민 → 교육 목적이라 ReLU 선택 |
| 인코더 블록 | `encoder.py` | MHA → Add&Norm → FFN → Add&Norm | Pre-LN 도 시험 예정, 1차는 Post-LN 유사 흐름 |
| 실행/검증 | `main.py` | 더미 입력 생성, shape/동작 검증 | 최소 seed 고정, mask 는 생략 |

**현우**: "이번엔 클래스 하나에 다 우겨넣지 않고, '학습 단위' 로 물리적으로 파일 나눈 게 좋았어."

**지영**: "각 파일 열어보며 연결 구조 추적하는 과정이 진짜 아키텍처 감 잡는 데 도움 됨." 

## 모듈 단위로 다시 보기

### 1) 임베딩 & 위치 인코딩 (`embedding.py`)

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

교육용 PoC이므로 변형 없이 구조 그대로를 구현하는데에 집중하였습니다.

### 2) Scaled Dot-Product & Multi-Head (`attention.py`)

```python
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V):
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attn = scores.softmax(dim=-1)
        return attn @ V, attn
```

핵심 체크 포인트:
1. shape 점검을 print 로 넣었다가 커밋에서 제거 (노이즈)
2. dropout 은 첫 버전 생략 → 이후 실험 TODO 로 README 에 남김
3. causal mask 미포함 (Encoder-only 흐름이므로)

### 3) MultiHeadAttention 단순화 결정

`view → transpose → matmul → concat → linear` 의 정석 흐름을 숨기지 않고 그대로 노출. 한 줄 최적화 대신 *학습 가시성* 우선.

**민수**: "einsum 버전이 더 간지나 보였는데, 디버깅 때는 verbose 형태가 확실히 편했음." 

### 4) Feed Forward (`feed_forward.py`)

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
```


**지영**: "linear1 뒤에 ReLU (또는 GELU) 안 넣고 '모델 표현력이 왜 이렇게 약하지?' 라고 묻는 경우 진짜 많았어."  
**민수**: "d_ff를 d_model이랑 똑같이 둬서 확폭 이점 자체를 못 느끼고 그냥 'Transformer 별로네' 라는 결론 내리기도 하고."  
**현우**: "dropout 위치 헷갈려서 ReLU 전에 넣거나 linear2 이후에만 두고 재현성 차이로 시간 날린 적 있었지 — 권장은 linear1 → 활성함수 → dropout → linear2 흐름."  
**지영**: "F.relu(..., inplace=True) 썼다가 residual 더할 때 원본 덮여서 미묘한 값 꼬이는 거 의심만 하다 시간 낭비하기도 하고."  
**민수**: "과적합만 무서워서 d_ff를 지나치게 줄여놓고는 '왜 학습이 안 오르지?' → 사실 capacity 부족인데 말이야."  
**현우**: "이 다섯 가지만 체크리스트로 돌려도 FFN 디버깅 시간 확 줄어."  

### 5) Encoder Block (`encoder.py`)

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x))
        x = self.norm2(x + self.ffn(x))
        return x
```

트랜스포머는 목적에 따라 인코더만을 사용하여 구현하기도 하고, PoC를 통한 교육 목적이므로 Encoder만 구현하여 실험하도록 지도하였습니다.

### 6) 실행/통합 (`main.py`)

```python
def run():
    batch, seq_len, vocab_size = 2, 16, 1000
    d_model, n_heads, d_ff, layers = 128, 4, 512, 2
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    emb = TokenEmbedding(vocab_size, d_model)(tokens)
    emb = PositionalEncoding(d_model)(emb)
    x = emb
    for _ in range(layers):
        x = EncoderBlock(d_model, n_heads, d_ff)(x)
    print('output shape:', x.shape)
```

마지막으로 하이퍼파라미터를 설정하고 직접 실행해보는 과정을 겪어보는 체험형 PoC를 겪어볼 수 있어요.

## 디자인 & 트레이드오프 로그

| 이슈 | 선택 | 대안 메모 |
|------|------|-----------|
| 활성함수 | ReLU | GELU 는 후속 (추적성 > 성능) |
| LayerNorm 위치 | Pre-LN | 학습 안정성, Post-LN 은 잔차 폭발 risk 데모 설명 어려움 |
| dropout | 생략 | determinism + 단순화, 교육 후 추가 계획 |
| mask | 없음 | Encoder-only + 고정 길이 실습 |
| 파라미터 스케일 | 소형 (≤ 1M) | CPU 에서 즉시 실행 검증 |

**현우**: "성능보다 '눈으로 구조 따라가기' 를 우선한 결정들이 일관돼서 좋았음." 

## 실험에서 얻은 미세한 인사이트

1. PositionalEncoding 을 매 step 재계산하지 않고 buffer 로 등록 → shape 버그 감소
2. layer 쌓기보다 d_model 을 늘릴 때 메모리/속도 비선형 증가 체감 → 추후 profiling 항목 추가
3. seed 고정이 없을 때 첫 attention head 시각화 값이 설명 자료랑 달라 팀 혼선 → `torch.manual_seed(42)` 도입
4. FFN d_ff 축소 시 정보 병목(activation variance 감소) 관찰 → histogram 로깅 필요성 느꼈음

## 만약 2차 버전을 만든다면 (Roadmap)

다음은 레포 README TODO 로도 이전 예정:

- [ ] Causal mask / padding mask 분리 구현
- [ ] Flash Attention(or scaled causal kernel) 비교 실험
- [ ] GELU + Dropout + Residual scaling (DeepNet) 옵션화
- [ ] 학습 루프 + 학습률 warmup / cosine 스케줄 예제 추가
- [ ] Benchmark: seq_len 증가에 따른 latency 테이블
- [ ] 작은 한국어 코퍼스 micro pretrain (subword tokenizer 포함)

**지영**: "TODO 리스트 자체가 이제 '어떻게 확장할지' 사고 프레임이 된 듯." 

## GPT 계열과의 구조적 동일성 관찰

| 항목 | 우리 PoC | GPT-2 Small (참고) |
|------|----------|--------------------|
| Layers | 2 | 12 |
| d_model | 128 | 768 |
| Heads | 4 | 12 |
| d_ff | 512 | 3072 |
| Positional | Sin/Cos | Learned (absolute) |
| Norm 위치 | Pre-LN | (변형 depending impl) |

크기 차이는 극단적이지만 조립 순서 · 서브레이어 패턴은 동일. “스케일이 complexity 를 의미하지 않는다” 체득.

**민수**: "결국 우리가 만진 이 작은 블록들의 반복이라는 거잖아." 

## 3개월 스터디 여정 돌아보기

### 우리가 배운 것들

각 주차별로 습득한 핵심 개념들:

"""
study_journey_recap

요약:
  8주(약 3개월) 동안 Transformer 핵심 구성요소를 단계적으로 학습한 흐름을
  일관된 출력 포맷으로 정리해 콘솔에 보여주는 유틸리티 함수.

기능:
  - 주차(1주차~8주차)를 키로, 각 주의 학습 항목을 값(사전)으로 가지는 구조 정의
    * 주제: 그 주의 메인 개념
    * 핵심: 한 줄 핵심 요약
    * 깨달음: 실천/이해 관점에서 얻은 통찰
  - 정의된 순서대로 콘솔에 사람이 읽기 쉬운 형식으로 출력

출력 형식 예:
  === 3개월 스터디 여정 ===

  1주차: Transformer 개요
    핵심: RNN의 한계와 Attention의 혁신
    깨달음: 병렬처리 + 장거리 의존성 해결
  ...
  8주차: 전체 구현
    핵심: 모든 퍼즐 조각 맞추기
    깨달음: 복잡해 보이는 것도 단순한 원리의 조합

사용 시나리오:
  - 블로그/노트 마무리 섹션 자동 생성
  - 주차별 커리큘럼 회고 출력
  - 추가 메타정보(예: 실습 코드 링크) 필드 확장 기반으로 손쉬운 발전

확장 아이디어:
  - weeks 사전을 외부 JSON/YAML로 분리해 재사용성 확보
  - 출력 포맷을 Markdown / HTML / CSV 등으로 선택 가능하게 일반화
  - CLI 옵션(--format, --filter-week 등) 추가
  - 정렬을 보장하기 위해 OrderedDict 혹은 리스트 활용 (파이썬 3.7+는 dict 삽입순서 유지)

주의:
  - 현재는 단순한 print 기반 (로깅/국제화 미포함)
  - 주차 키/필드명이 하드코딩되어 있어 스키마 변경 시 함수 수정 필요

요약 한줄:
  "Transformer 스터디의 주차별 핵심 개념과 통찰을 구조화하여 빠르게 회고할 수 있게 하는 출력 도우미."
"""
### 3개월 스터디 여정

#### 1주차: Transformer 개요
- 핵심: RNN의 한계와 Attention의 혁신
- 깨달음: 병렬처리 + 장거리 의존성 해결

#### 2주차: Positional Encoding
- 핵심: 순서 없이 순서 기억하기
- 깨달음: 수학으로 위치 정보 인코딩

#### 3주차: Feed Forward Network
- 핵심: 정보 변환의 실제 엔진
- 깨달음: 비선형성으로 복잡한 패턴 학습

#### 4주차: Residual Connection & Layer Norm
- 핵심: 깊은 네트워크를 가능하게 하는 기반
- 깨달음: 안정성이 성능의 전제조건

#### 5주차: Self-Attention 수학
- 핵심: Query, Key, Value의 진실
- 깨달음: 벡터 내적으로 의미 유사도 측정

#### 6주차: Multi-Head Attention
- 핵심: 다양한 관점으로 보기
- 깨달음: 여러 전문가의 협업이 더 강력

#### 7주차: 임베딩과 토크나이저
- 핵심: 단어를 벡터로 변환하기
- 깨달음: 입력 품질이 전체 성능 결정

#### 8주차: 전체 구현
- 핵심: 모든 퍼즐 조각 맞추기
- 깨달음: 복잡해 보이는 것도 단순한 원리의 조합

### 스터디에서 가장 기억에 남는 순간들

**각자의 "아하!" 순간:**

> **민수**: "Positional Encoding에서 더하기만으로 위치 정보가 전달된다는 걸 실험으로 확인했을 때"
> 
> **지영**: "Multi-Head Attention에서 각 헤드가 자동으로 다른 역할을 학습한다는 걸 알았을 때"
> 
> **현우**: "벡터 내적이 실제로 의미적 유사도와 일치한다는 걸 수치로 확인했을 때"

## 앞으로의 학습 방향

### 다음 단계 추천

Transformer를 마스터한 후 나아갈 방향들:

### 다음 단계 추천

1. 좋은 자료: LLM Paper Learning 페이지 참고하기.
[LLM Paper Learning](https://github.com/PasaLab/LLM_Paper_Learning)
2. 아키텍쳐 학습: 다양한 Transformer 변형 및 응용 사례 탐색
3. 최적화 기법: 메모리 및 속도 개선을 위한 다양한 기법 실험
4. 실무 적용: 실제 프로젝트에 Transformer 모델 통합 및 최적화
5. 최신 연구 동향: Transformer 관련 최신 논문 및 기술 동향 파악

## 마지막 메시지

드디어 8편의 대장정이 끝났습니다. 처음엔 막막해 보였던 "Attention Is All You Need" 논문이 이제는 친숙하게 느껴지시나요?

저는 스터디원들이 제 스터디에서 Transformer 구조 하나만큼은 꼭 아이디어를 얻어가길 바라며 스터디를 진행했고, 전반적으로 잘 진행해주었기 때문에 매우 뿌듯합니다. 

이론적 기반이 모두 준비되어있는 학생들은 적었지만, 좋은 직관을 갖고 빠르게 공부하는 학생들이 있어 참여가 활발하였습니다.

### 핵심 메시지들

**3개월간 얻은 가장 중요한 깨달음들:**

1. **복잡함 속의 단순함**: Transformer는 복잡해 보이지만 각 구성요소는 명확한 역할이 있습니다.

2. **수학의 아름다움**: 벡터 내적, 소프트맥스, 층 정규화 등 모든 수학적 요소에는 분명한 이유가 있어요.

3. **협업의 힘**: Multi-Head처럼 여러 관점을 조합하면 더 강력한 결과를 얻을 수 있습니다.

4. **기초의 중요성**: 토크나이저나 임베딩 같은 입력 처리가 전체 성능을 좌우합니다.

5. **실험의 가치**: 이론만으로는 부족하고, 직접 코드를 짜보고 실험해봐야 진정한 이해가 가능합니다.

### 마지막 당부

ChatGPT를 쓸 때마다, 새로운 AI 뉴스를 볼 때마다, 그 뒤에 숨겨진 원리들이 보일 겁니다. Self-Attention이 어떻게 문맥을 파악하는지, Multi-Head가 왜 필요한지, 토크나이저가 성능에 어떤 영향을 주는지.

**다음에 또 다른 혁신적인 아키텍처가 나온다면, 우리는 두려워하지 않고 그 원리를 파헤쳐볼 수 있을 거예요.**

---

*P.S. 3개월이라는 시간이 길게 느껴졌지만, 돌이켜보니 정말 알찬 여정이었습니다. 함께 공부하고, 토론하고, 실험했던 모든 순간들이 소중한 자산이 되었어요.*

**Transformer 완전정복, 축하합니다! 이제 여러분은 AI의 언어를 구사할 수 있습니다.**