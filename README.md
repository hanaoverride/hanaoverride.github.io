# hanaoverride.github.io

개인 기술 & 학습 노트 블로그 저장소입니다. GitHub Pages + Jekyll 로 동작합니다.

## ✅ 개요
- 주소: https://hanaoverride.github.io
- 목적: LLM, 컴퓨터과학, 보안, 프로젝트 기록 모음
- 스택: GitHub Pages (Jekyll, `github-pages` gem), 테마: minima (커스터마이징 예정)

## 🗂 폴더 구조
```
_config.yml        # Jekyll 설정
_pages/            # 카테고리별 글 (현재 post 형태 레이아웃 사용)
  categories/
    <category>/
      YYYY-MM-DD-title.md
LICENSE-CC-BY-SA   # 글 기본 라이선스
Gemfile            # Gem 의존성 (github-pages)
```
> `_posts/` 대신 `_pages/categories/<카테고리>/` 구조를 사용 중. 파일명에 날짜를 넣어 퍼머링크와 정렬에 활용.

## 🧾 글 작성 규칙
1. 파일 위치: `_pages/categories/<카테고리>/YYYY-MM-DD-title.md`
2. 파일명 규칙: `YYYY-MM-DD-슬러그.md` (슬러그는 소문자/하이픈)
3. Front Matter 예시:
   ```yaml
   ---
   layout: post
   title: "Attention Is All You Need 논문 : 처음 읽어보기! - (1)"
   date: 2025-02-20 18:20:24 +0900
   categories: [llm-engineering]
   tags: [Transformer, 논문리뷰]
   ---
   ```
4. `categories` 는 배열 형태. 현재 1개만 넣어도 배열 유지.
5. 이미지: `assets/images/<카테고리>/...` (폴더 아직 없으면 생성)
6. 수식: kramdown + MathJax (원하면 `_includes/head.html` 커스터마이징으로 추가 예정)

## 🛠 로컬 개발 환경 (Windows 기준)
### 1. Ruby 설치
- https://rubyinstaller.org 에서 Ruby+Devkit 최신 LTS 설치 (예: 3.x)
- 설치 후 PowerShell 새 창 열기

### 2. 저장소 클론
```powershell
git clone https://github.com/hanaoverride/hanaoverride.github.io.git
cd hanaoverride.github.io
```

### 3. Bundler & 의존성 설치
```powershell
gem install bundler
bundle install
```

### 4. 로컬 서버 실행
```powershell
bundle exec jekyll serve --livereload
```
기본 주소: http://127.0.0.1:4000 (또는 http://localhost:4000)

변경 즉시 자동 리빌드. 에러 발생 시 메시지를 그대로 검색하면 대부분 해결 가능.

## 🔁 배포 (GitHub Pages)
- `main` 브랜치에 push 하면 GitHub Pages 가 자동 빌드 & 배포.
- 상태 확인: 저장소 Settings > Pages > Build and deployment 로그
- 실패시 확인 포인트:
  - `_config.yml` 에 지원 안 되는 플러그인 추가했는지
  - YAML Front Matter 구문 오류 (탭 대신 공백 사용)

## 🔍 퍼머링크 & URL
`permalink: /:categories/:year/:month/:day/:title/` 형태.
예: `_pages/categories/llm-engineering/2025-02-20-attention-is-all-you-need-part1.md`
→ `/llm-engineering/2025/02/20/attention-is-all-you-need-part1/`

## 🧩 메타/SEO
- `jekyll-seo-tag` 플러그인 사용 → `<head>` 자동 메타 삽입
- RSS/Atom 피드: `/feed.xml`
- 사이트맵: `/sitemap.xml`

## 🪪 라이선스
`LICENSE-CC-BY-SA` (CC BY-SA 4.0). 글/이미지 활용 시 출처 표기 & 동일조건변경허락.

## ❓ Troubleshooting
| 증상 | 원인 | 해결 |
|------|------|------|
| `Could not locate Gemfile` | 경로 잘못됨 | `ls` 로 파일 존재 확인 후 이동 |
| `jekyll command not found` | bundle exec 미사용 | `bundle exec jekyll serve` 로 실행 |
| 빌드는 되는데 스타일 깨짐 | 캐시/브라우저 문제 | 강력 새로고침 (Ctrl+F5) |
| Liquid syntax error | `{% %}` `{ { } }` 오타 | 해당 파일 줄번호 확인 후 수정 |

## 🎨 현재 테마 (Hacker) & 커스터마이징
현재 테마: **jekyll-theme-hacker** (GitHub Pages 기본 호환). 추가 스타일은 `assets/css/style.scss` 에서 오버라이드.

커스터마이징 방법:
1. 전역 폰트/레이아웃: `assets/css/style.scss` 수정
2. 다크톤/포인트 컬러 변경: 링크/버튼 관련 `a`, `.tags a` 등 선택자 수정
3. 별도 컴포넌트 (배지, 경고 박스 등) 추가 시:
  ```scss
  .note-box { border-left:4px solid #0f0; background:#101810; padding:.75rem 1rem; margin:1.5rem 0; }
  .warn-box { border-left:4px solid #f90; background:#181310; }
  ```
4. 코드 하이라이트 테마 변경: Rouge 테마 → `_sass` 에 커스텀 스타일 추가 가능 (필요 시 안내 요청!)
5. favicon / OG 이미지: `assets/images/` 에 추가 후 `_includes/head` 커스터마이즈.

테마를 다른 것으로 바꾸려면 `_config.yml` 의 `theme:` 값을 변경한 뒤 `bundle update`.

## 🧱 향후 개선 아이디어
- 태그/카테고리 인덱스 자동 생성 스크립트
- 다크모드 토글
- Mermaid 다이어그램 지원 (`mermaid.js` include)
- 검색 (lunr.js 또는 simple-jekyll-search)

## 🤝 기여
PR 및 Issue 환영. 구조 단순 유지 지향.

---
문의: hanaoverride@gmail.com / https://github.com/hanaoverride
