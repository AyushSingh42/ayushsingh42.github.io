---
layout: home
title: ""
lead: "CS + Linguistics @ UIUC"
---

I study computer science and linguistics at the University of Illinois Urbana-Champaign. My interests include natural language processing, computational linguistics, and the study of language through data-driven and formal methods.

I am particularly interested in mechanistic interpretability of neural language models, with an emphasis on understanding how linguistic structure and reasoning emerge in learned representations. My work explores probing tasks, representation analysis, and causal interventions as tools for studying internal model behavior.

Outside of academics, I enjoy reading, watching and analyzing basketball, and writing informally. This site serves as a place to organize research notes, longer essays, and thoughts that do not fit neatly elsewhere.

## Recent writing
{% for post in site.posts limit:4 %}
- [{{ post.title }}]({{ post.url | relative_url }}) <span class="small">{{ post.date | date: "%b %d, %Y" }}</span>
{% endfor %}
