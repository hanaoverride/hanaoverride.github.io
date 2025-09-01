---
layout: post
title: "왜 LLM API는 매번 대화 기록을 다시 보낼까: Transformer의 무상태성과 컨텍스트 관리의 필연성"
date: 2025-08-31 10:12:34 +0900
categories: [llm-engineering]
tags: [LLM, API설계, Stateless, 세션관리, 아키텍처, Transformer, Attention, Context-Window]
redirect_from:
  - /llm-engineering/2025/08/31/llm-api-stateless/
---

# 왜 LLM API는 매번 대화 기록을 다시 보낼까: Transformer의 무상태성과 컨텍스트 관리의 필연성

저는 Openrouter를 사용해 챗봇 서비스를 개발하는 도중, 라우터가 프로바이더를 변경할 경우 어떻게 캐시와 컨텍스트가 유지되는지 궁금하더군요. OpenRouter는 요청마다 가장 저렴하고 빠른 프로바이더를 자동으로 선택한다고 하는데, GPT에서 Claude로 바뀌어도 대화가 자연스럽게 이어집니다. 어떻게 가능한 걸까요?

사실 원리는 간단합니다!: **매번 전체 대화 기록을 클라이언트가 다시 보내고 있었습니다.**

이 방식은 겉보기에는 매우 비효율적으로 보입니다. 매번 텍스트가 길어질수록 토큰 비용이 늘어나고, 네트워크 대역폭도 낭비되죠. 뭔가 서버가 상태를 기억해 줄수는 없는걸까? 라는 생각이 들기도 하고요.

오늘은 이 문제를 파고들어봤습니다. 왜 2025년에도 우리는 이런 방식을 쓰고 있을까요? 그리고 현업에서는 이걸 어떻게 해결하고 있을까요?

## OpenRouter는 아무것도 저장하지 않는다

OpenRouter의 경우 openai 패키지를 그대로 사용하므로, 구현체 역시 openai의 방법을 그대로 사용하면 됩니다. 

"세션" 대화는 보통 고전적으로는 chatcompletion을 사용하거나, 최근 정식 기능이 된 conversations API를 사용하면 되는데요, 여기서는 chatcompletion 방식을 사용한 예제를 보겠습니다.

```python
import openai
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("OPENROUTER_API_KEY")

# OpenRouter 설정
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=key
)

# 첫 번째 요청 (GPT로 라우팅될 수 있음)
response1 = client.chat.completions.create(
    model="openrouter/auto",  # 자동 프로바이더 선택
    messages=[
        {"role": "user", "content": "내 이름은 철수야"}
    ]
)
# 응답: "안녕하세요 철수님!"

# 두 번째 요청 (Claude로 라우팅될 수 있음)
response2 = client.chat.completions.create(
    model="openrouter/auto",
    messages=[
        {"role": "user", "content": "내 이름은 철수야"},
        {"role": "assistant", "content": "안녕하세요 철수님!"},
        {"role": "user", "content": "내 이름이 뭐라고?"}  # 새 질문
    ]
)
# 응답: "철수님이라고 말씀하셨습니다."
```

결론적으로 구현 방식을 따라가다 보면 OpenRouter는 프로바이더 간 전환 시 아무것도 저장하지 않는다는 사실을 알 수 있습니다. 

대신 클라이언트가 매번 전체 대화 히스토리를 포함시켜 보내기 때문에, 어떤 프로바이더로 바뀌어도 전체 컨텍스트를 받아 자연스럽게 응답할 수 있는 것입니다.

### Microsoft의 공식 가이드라인

이는 OpenRouter만의 특이한 구현이 아닙니다. Microsoft의 Azure OpenAI 문서에서도 `"Because the model has no memory, you need to send an updated transcript with each new question or the model will lose the context of the previous questions and answers"`라고 명시하고 있습니다.

> 원문: [Work with the Chat Completions models](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/chatgpt?pivots=programming-language-chat-completions#working-with-the-chat-completion-api)


```python
import os
from openai import AzureOpenAI

client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2024-10-21",
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # Your Azure OpenAI resource's endpoint value.
)

conversation=[{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_input = input("Q:")      
    conversation.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o", # model = "deployment_name".
        messages=conversation
    )

    conversation.append({"role": "assistant", "content": response.choices[0].message.content})
    print("\n" + response.choices[0].message.content + "\n")
```

## 기존 웹 채팅 서비스와의 충돌: 왜 이렇게 다른가?

### 전통적인 웹 채팅 구현: 세션 기반 구현체

우리가 웹 기반 채팅 서비스를 구현한다면 보통 어떻게 구현할까요? 우선 **"세션"** 을 만들어 서버가 상태를 기억하도록 하겠죠? 그리고는 클라이언트가 일정 시간동안 응답하지 않으면 그 세션을 만료시키는겁니다. 대화 내용은 서버에 저장할수도, 저장하지 않을수도 있지만 LLM 서비스처럼 대화 한번에 전체 히스토리가 오가는 방식은 드물죠. 실제로도 socket.io의 공식 문서를 보면 다음과 같은 내용을 언급하고 있어요:

The server sends a ping packet every `pingInterval` ms, and if the client does not answer with a pong within `pingTimeout` ms, the server considers that the connection is closed.

ChatGPT와 같은 서비스도 "세션"이라는걸 제공하긴 하죠? 하지만 일반적으로 컴퓨터공학에서 말하는 세션처럼 서버가 메모리 상에 상태를 유지하는 방식은 아닌 것 같습니다. 웹에서 언제든지 접근할 수 있잖아요?

전통적인 "세션" 기반 채팅 서비스 구현에 익숙했던 개발자라면 오히려 왜 이런 식으로 multi-round conversation이 구현되는지 의아할 수밖에 없습니다. 일종의 지식의 저주라고 할 수 있겠네요.

## Transformer의 본질적 한계: 왜 Stateless일 수밖에 없는가

### Attention 메커니즘의 작동 원리

2017년 "Attention Is All You Need" 논문에서 제안된 Transformer 아키텍처는 self-attention 메커니즘을 핵심으로 합니다.

> self-attention 메카니즘에 대한 지식이 부족하다면 우선적으로 Attention is All You Need 논문을 읽어보시길 권장합니다.: [Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)
> 만약 이 논문이 이해하기 어렵다면 제가 8회차에 걸쳐 해당 논문을 스터디했던 내용을 참고하면 도움이 될 수 있을거에요.: [Attention is All You Need 스터디 노트](https://hanaoverride.github.io/llm-engineering/2025/08/29/transformer-study-part1-overview/)

Self-attention 메커니즘은 입력 시퀀스의 모든 위치가 다른 모든 위치를 동적으로 참조할 수 있게 하지만, 이는 현재 입력 내에서만 가능합니다. **즉, 이전 요청의 hidden state를 유지하거나 참조하는 기능이 없습니다.** 이는 Transformer가 본질적으로 stateless한 아키텍처임을 의미합니다.

한편 여러분이 네트워크 지식이 좀 있다면, HTTP 또한 stateless하다는 점을 알고 계실겁니다. stateless한 모델이 stateless한 프로토콜 위에서 동작하는 셈이니 뭔가 아주 특별한 이유가 있는게 아니라면 일관적으로 stateless한 파이프라인에 state를 유지해주려는 시도가 필요할 것 같진 않네요.

## 시도해 볼 수 있는 최적화 아이디어?

여러 가지 생각을 해 보았으나 사실 굳이 적용해야 하나는 좀 의문이라는 생각이 들었습니다. 그래도 우선 고민하는 과정에서도 얻을 게 있으니, 몇 가지 아이디어를 생각해 볼까요?

### Idea 1: 어차피 긴 컨텍스트를 다 못 읽으면 잘라서 읽으면 어떨까?

#### 배경 지식: "Lost in the Middle" 현상

Liu et al. (2023)의 연구에 따르면, LLM은 긴 컨텍스트에서 정보의 위치에 따라 성능이 크게 달라집니다. 특히 관련 정보가 입력의 시작이나 끝에 있을 때 성능이 가장 높고, 중간에 있을 때 현저히 떨어집니다.

원문을 인용하자면 `"prompting language models with longer input contexts is a trade-off—providing the language model with more information may help it perform the downstream task, but it also increases the amount of content that the model must reason over, potentially decreasing accuracy"`라는 관찰인데요, 

최근 모델에서 이런 문제들을 해결하려는 시도가 많이 보이지만 여전히 모델이 제공하는 최대 컨텍스트인 100만에 비해 10만 정도에서 성능 저하가 관찰된다는 커뮤니티 피드백이 많은 것을 보면, 아직까지 완전히 해결하기는 요원해 보입니다.

#### 절사 후 전송 전략

바로 LLM이 이렇게 긴 컨텍스트를 어차피 제대로 처리하지 못 할 것이라는 점 때문에, 최근 정보만 절사하여 전송하는 식으로 최적화를 시도해 볼 수 있겠습니다.

```python
import openai

# ... previous code ...

response = client.chat.completions.create(
    model="gpt-5",
    messages=messages[-10:]  # 최근 10개의 메시지만 전송
)
```

하지만 대부분의 LLM 프로바이더의 경우 "캐시"를 제공합니다. 이미 사용한 부분에 대해서 캐시에 들어있는 정보는 원래 비용에 비해 대폭 할인을 해주는 방식인데요, cache hit를 할 경우 알아서 비용이 절감되기 때문에 굳이 히스토리를 절사해서 보내는 식으로 최적화를 해야할 지는 좀 의문이긴 합니다.(제공자 정책에 따라 다르지만, 절사한 프롬프트에도 cache hit가 적용되므로 여전히 비용을 절약하는 의의가 있겠지만요)

또한 제일 먼저 제시한 정보라도 대화에 관련이 있을 수 있는데 너무 단순하게 정보를 절사하여 맥락을 잃을 수 있다는 약점이 있겠네요.

### Idea 2: RAG로 관련 대화만 선택적으로 포함시키기

#### 배경 지식: RAG (Retrieval-Augmented Generation)

RAG는 벡터 데이터베이스에 저장된 관련 정보를 검색하여 LLM의 입력에 포함시키는 기법입니다. 이를 통해 모델에 관련성 있고 참조하고 싶은 정보를 응답에 포함하도록 유도할 수 있습니다.

주로 문장을 임베딩이라는 벡터로 변환한 후, 코사인 유사도 혹은 유클리드 거리 등의 기준을 사용하여 
"의미 유사도가 높은" 문서들을 가져오는 방식으로 사용합니다. 인터넷 문서를 가져오거나, 사용자 대화를 분석하여 최적화된 답변을 생성하는 등의 용도로 사용할 수 있어요.

#### RAG 기반 히스토리 선택

바로 이 RAG 기술을 히스토리의 선택적 전송에 이용해보는 아이디어는 어떨까요? 예를 들어 현재 보내려는 사용자 쿼리 중에서, 과거 대화에서는 크게 관련이 없는 부분들이 있을 수 있습니다. 필요한 부분만 보내서 보낸다면 비용 절감에 도움이 되면서, 실제로 관련 있는 정보만 보낼 수 있곘죠.

```python
import openai
from pinecone import Pinecone, Index

# ... previous code ...

index = text_index: Index = pc.Index(settings.INDEX_NAME)
response = openai.embeddings.create(
        input=topic,
        model="text-embedding-3-small"
    )
query_embedding = response.data[0].embedding

results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)
```

또한 RAG 기술을 사용함으로서 관련있는 정보만 모아서 전송할 수 있다는 장점 이외에도, 다른 "세션"의 정보를 참조하거나, 더 참조시키고 싶은 외부 문서를 함께 포함시킬 수도 있다는 장점도 있습니다.


## 되돌아보기: 왜 LLM API는 왜 모든 대화 기록을 다시 보내는가

best practice와 이론적 배경을 알아보고, 개선 방법까지 고민해봤으면 이제 다시 읽어본 내용을 되돌아봅시다.

1. **Stateless 현실**: 서버(프로바이더)는 대화 상태를 안 들고, 클라이언트가 전체 히스토리를 재전송(OpenRouter/Azure 사례).  
2. **구조적 원인**: Transformer self-attention은 호출 간 hidden state 미보존 → 필요한 과거는 항상 프롬프트에 포함.  
3. **주요 최적화 레버**: (a) 부분 절사 (b) RAG 선택 삽입 (c) 프롬프트 캐시로 비용 절감.  

### 맺음말

때로 우리는 문제에 대해 `잘못된 해결책`을 생각하며, 그 해결책이 왜 구현되지 않는지 고민하곤 합니다. 애초에 그것은 `필요 없는 것`임에도 불구하고!

저는 `왜 모델이 stateful하지 않은지` 생각하며 문제를 해결하려 했지만, 애초에 `잘못된 해결책을 상정`하고 있었다는 결론을 내렸습니다. 실제로는 모델이 ‘기억해주길’ 바라며 상태를 붙잡으려 하기보다 stateless 전제를 받아들이고 요약, RAG, 캐시로 매 턴 맥락을 재조립하는 설계가 비용, 확장성, 단순성을 동시에 잡을 수 있을 것이라는 생각을 했습니다.

트위터에서 이런 글을 본 적이 있습니다. `"When life gives you lemons, make lemonade."`라는 말은 주로 인생의 고난을 긍정적으로 받아들이라는 의미로 쓰이지만, 실제로는 레몬이 주어진다면 레모네이드를 만드는 것이 가장 합리적인 선택이라는 뜻이기도 하죠.

마찬가지로 stateless한 모델이 주어진다면, **그 제약을 받아들이고 그 안에서 최적의 설계를 고민하는 것이 가장 합리적인 선택일 것입니다.**

---

## 참고문헌

1. Microsoft Azure OpenAI Documentation - Working with the Chat Completion API
2. Vaswani et al. (2017). "Attention Is All You Need". arXiv:1706.03762
3. Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts". arXiv:2307.03172
4. OpenRouter API Documentation