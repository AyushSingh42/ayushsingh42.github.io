---
layout: home
title: "Home"
lead: "CS + Linguistics @ UIUC"
---

I study computer science and linguistics at the University of Illinois Urbana-Champaign. My interests include natural language processing, computational linguistics, and the study of language through data-driven and formal methods.

I am particularly interested in mechanistic interpretability of neural language models, with an emphasis on understanding how linguistic structure and reasoning emerge in learned representations. My work explores probing tasks, representation analysis, and causal interventions as tools for studying internal model behavior.

Outside of school, I like to read, watch lots and lots of basketball, and watch movies (and sometimes I write about them which you can find [here](https://ayushsingh42.substack.com/?utm_campaign=profile_chips)). When I am not inside I like to play basketball and go bouldering. 

## News

{% assign all_news = site.data.news | sort: 'date' | reverse %}
{% for item in all_news limit: 5 %}
- <span class="small">{{ item.date | date: "%b %d, %Y" }}</span> — {{ item.content | markdownify | remove: '<p>' | remove: '</p>' }}
{% endfor %}