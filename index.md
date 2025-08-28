---
layout: default
title: 홈
---

# hanaoverride's notebook

> LLM · 컴퓨터과학 · 보안 · 글쓰기 · 프로젝트 기록

최근 작성한 글 (최신순):

<ul>
{%- assign allpages = site.pages | where_exp: "p","p.path contains '/_pages/categories/'" -%}
{%- assign posts_sorted = allpages | sort: "date" | reverse -%}
{%- for p in posts_sorted limit: 15 -%}
  <li>
    <span style="white-space:nowrap;font-variant-numeric:tabular-nums;color:#6f6;">{{ p.date | date: '%Y-%m-%d' }}</span>
    <a href="{{ p.url }}">{{ p.title }}</a>
    {%- if p.categories and p.categories.size > 0 -%}
      <span style="font-size:0.75rem;opacity:.7;">[{{ p.categories | join: ', ' }}]</span>
    {%- endif -%}
  </li>
{%- endfor -%}
</ul>

[전체 글 보기 →](/)

---

## 카테고리
<ul style="columns:2; -webkit-columns:2; -moz-columns:2; list-style:none; padding-left:0;">
{%- assign catset = "" | split: "" -%}
{%- for p in allpages -%}
  {%- for c in p.categories -%}
    {%- assign catset = catset | push: c -%}
  {%- endfor -%}
{%- endfor -%}
{%- assign uniquecats = catset | uniq | sort -%}
{%- for c in uniquecats -%}
  <li><a href="/{{ c }}/">{{ c }}</a></li>
{%- endfor -%}
</ul>

---

<p style="font-size:0.75rem;opacity:.6;">Powered by Jekyll Hacker theme · Custom index</p>
