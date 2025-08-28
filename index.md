---
layout: default
title: 홈
paginate: 15
---

# hanaoverride's notebook

> LLM · 컴퓨터과학 · 보안 · 글쓰기 · 프로젝트 기록

<ul>
{%- comment -%}
If jekyll-paginate enabled (paginate in _config.yml), paginator.posts exists.
Fallback to site.posts when not.
{%- endcomment -%}
{%- assign collection = paginator.posts | default: site.posts -%}
{%- for p in collection -%}
  <li>
    <span style="white-space:nowrap;font-variant-numeric:tabular-nums;color:#6f6;">{{ p.date | date: '%Y-%m-%d' }}</span>
    <a href="{{ p.url }}">{{ p.title }}</a>
    {%- if p.categories and p.categories.size > 0 -%}
      <span style="font-size:0.7rem;opacity:.65;">[{{ p.categories | join: ', ' }}]</span>
    {%- endif -%}
  </li>
{%- endfor -%}
</ul>

{%- if paginator.total_pages and paginator.total_pages > 1 -%}
<nav class="pager" style="margin-top:1.5rem;">
  {%- if paginator.previous_page -%}
    <a href="{{ paginator.previous_page_path }}">← 이전</a>
  {%- endif -%}
  <span style="margin:0 1rem;">Page {{ paginator.page }} / {{ paginator.total_pages }}</span>
  {%- if paginator.next_page -%}
    <a href="{{ paginator.next_page_path }}">다음 →</a>
  {%- endif -%}
</nav>
{%- endif -%}

---

## 카테고리
<ul style="columns:2; -webkit-columns:2; -moz-columns:2; list-style:none; padding-left:0;">
{%- assign catset = "" | split: "" -%}
{%- for p in site.posts -%}
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

<p style="font-size:0.75rem;opacity:.6;">Powered by Jekyll Hacker theme · Posts index</p>
