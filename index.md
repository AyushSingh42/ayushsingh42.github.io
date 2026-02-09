---
layout: home
title: ""
lead: "CS + Linguistics @ UIUC"
---

I study computer science and linguistics at the University of Illinois Urbana-Champaign. My interests include natural language processing, computational linguistics, and the study of language through data-driven and formal methods.

I am particularly interested in mechanistic interpretability of neural language models, with an emphasis on understanding how linguistic structure and reasoning emerge in learned representations. My work explores probing tasks, representation analysis, and causal interventions as tools for studying internal model behavior.

Outside of academics, I enjoy reading, watching and analyzing basketball, and writing informally. This site serves as a place to organize research notes, longer essays, and thoughts that do not fit neatly elsewhere.

## News

{% assign all_news = site.data.news | concat: site.posts | sort: 'date' | reverse %}
{% for item in all_news limit: 5 %}
- <span class="small">{{ item.date | date: "%b %d, %Y" }}</span> — {% if item.title %}[Blog: {{ item.title }}]({{ item.url | relative_url }}){% else %}{{ item.content | markdownify | remove: '<p>' | remove: '</p>' }}{% endif %}
{% endfor %}
