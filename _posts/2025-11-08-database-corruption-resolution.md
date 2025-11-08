---
layout: post
title: "OpenWebUI 데이터베이스 커럽션 복구기"
date: 2025-11-08 18:37:12 +0900
categories: [self-hosting]
tags: [OpenWebUI, SQLite, PostgreSQL, 백업]
---

나는 홈 서버에서 개인 챗봇을 사용하기 위해 OpenWebUI 인스턴스를 돌리고 있다. (OpenWebUI는 ChatGPT 같은 서비스를 개인이 언어모델을 호출해서 사용할 수 있게 해주는 프로그램이다.) 몇 개월간 문제없이 쓰고 있었는데, 어제 갑자기 DB 커럽션이 발생해서 서비스 장애가 생겼다. 다행히 문제 파악 후 30분 만에 복구했고, 이 과정을 정리해봤다.

이전에도 단일 채팅 세션에서 커럽션이 일어난 적이 있었다. GitHub issues에도 보고된 백엔드의 채팅 세션 validation 관련 문제였는데 (https://github.com/open-webui/open-webui/issues/15189), 이번에도 비슷한 문제라고 생각했다. 근데 이번엔 채팅 전체 히스토리를 열람할 수 없게 되는 게 달랐다. 원인을 하나씩 생각해보기로 했다.

### 원인 규명

**SQLite3 자체의 커럽션 취약성?**

OpenWebUI는 기본적으로 SQLite3을 RDB로 사용한다. 처음엔 SQLite3 자체가 커럽션에 취약할 수도 있다고 의심했다. SQLite3는 네트워크 환경보다는 임베디드나 온디바이스에서 사용하기 적합한 RDBMS니까.

대부분의 엔터프라이즈급 RDBMS는 **트랜잭셔널**이라는 원자 단위 DB 작업 수행 기능을 제공한다. CRUD의 한 단위 작업을 하다가 중단되면 전체 작업을 하지 않은 것으로 처리해서, 일부만 데이터베이스 작업이 되어 무결성을 망치는 문제를 방지하는 거다. 이게 SQLite3엔 부족하지 않을까 싶었는데, 공식 문서를 보니까 분명히 지원하고 있었다.

https://sqlite.org/transactional.html
```markdown
SQLite is Transactional

A transactional database is one in which all changes and queries appear to be Atomic, Consistent, Isolated, and Durable (ACID). SQLite implements serializable transactions that are atomic, consistent, isolated, and durable, even if the transaction is interrupted by a program crash, an operating system crash, or a power failure to the computer.

We here restate and amplify the previous sentence for emphasis: All changes within a single transaction in SQLite either occur completely or not at all, even if the act of writing the change out to the disk is interrupted by

- a program crash,
- an operating system crash, or
- a power failure.
The claim of the previous paragraph is extensively checked in the SQLite regression test suite using a special test harness that simulates the effects on a database file of operating system crashes and power failures.
```

**SQLite3의 동시성 처리 문제**

두 번째로 의심한 건 SQLite3의 **동시성 처리 문제**였다. 사용자가 3명이라 크게 신경 안 썼는데, SQLite3은 다수 사용자가 동시에 쓰기를 하는 걸 고려하고 만든 RDBMS가 아니다. 동시에 쓰기를 요청하면 문제가 생길 수 있다.

```markdown
**Multiple** processes can have the same database **open** at the same time. Multiple processes can be doing a SELECT at the same time. But **only one** process can be **making changes** to the database at any moment in time, however.
```

실제로 많은 대기업이 SQLite3을 사용할 정도로 우수한 기술이긴 한데, **읽기 위주 환경이나 임베디드, 온디바이스 위주 사용을 상정하고 만든 프로그램**이다 보니 네트워크 다중 쓰기에서 문제가 생길 여지가 있다. 문제가 생긴 시간에 나랑 같은 시간에 작업하던 사람은 없었지만, 동시에 채팅하던 중 쓰기 오류가 누적되어 커럽션으로 이어졌을 가능성은 있어 보였다.

**OpenWebUI 백엔드의 무결성 체크 부족**

세 번째는 OpenWebUI 백엔드의 **무결성 체크 기능이 예전부터 부족**하다는 게 GitHub issues에서 계속 지적되고 있던 점이었다. 잘못된 응답이 오더라도 그걸 검증하지 않고 DB에 삽입하게 되면 그 자체가 DB 커럽션을 유발하는 거니까. 이 문제는 내가 코드를 직접 수정하지 않는 이상 해결할 수 없는 부분이다.

### 결론

결국 1, 2, 3이 모두 복합적으로 작용한 것 같다. 네트워크 동시 쓰기에 부적합한 SQLite3을 사용했고, OpenWebUI의 무결성 체크 부족으로 커럽션 데이터가 지속적으로 쌓여서 채팅 로그 전체가 먹통이 된 거다.

### 복구 과정

우선 12시간 단위로 BackBlaze B2 백업을 하고 있었기 때문에 DB 백업을 가져오는 걸 고려했다. 근데 바로 직전에 중요한 작업을 하고 있었어서 DB를 즉시 백업할 수 있는 옵션이 있는지 체크해봤다.

다행히 DB 파일 자체는 정상적으로 접근 가능해서 파일 손상은 아니라는 걸 확인했고, **SQLite 홈페이지의 Best Practice를 따라 커럽션 제거를 시도했다.**

https://sqlite.org/recovery.html
```bash
# 복구 SQL 파일 생성
sqlite3 webui.db ".recover" > recovered.sql

# 새로운 데이터베이스 생성
sqlite3 webui_new.db < recovered.sql
```

이후 새롭게 생성된 DB 파일을 컨테이너에 다시 넣었더니 아무 문제 없이 서버가 복구됐다.

### 앞으로의 계획

앞서 분석한 문제점들을 고려하면, SQLite3을 계속 사용할 경우 동일한 문제를 또 겪을 가능성이 높아 보인다. **다행히 OpenWebUI는 PostgreSQL을 지원하고 마이그레이션 기능도 제공하니까, PostgreSQL로 마이그레이션해서 네트워크 환경에 더 적합하게 만들 생각이다.**

백엔드 무결성 체크 기능은... 내가 시간이 많다면 직접 커뮤니티에 수정사항을 제안하고 싶은데, 할 일이 많아서 어렵겠다. 뭐 오픈소스 커뮤니티가 무료로 제공해주는 서비스이기도 하고, 여차하면 커럽션 제거 기능을 PostgreSQL에서도 동일하게 사용하면 될 것 같다.

그리고 기존 12시간마다 복구하던 걸 6시간마다로 좀 더 촘촘하게 바꿨다.