---
layout: home
title: "Home"
lead: "CS + Linguistics @ UIUC"
---

I study computer science and linguistics at the University of Illinois Urbana-Champaign. My interests include natural language processing, computational linguistics, and the study of language through data-driven and formal methods.

I am broadly interested in understanding and improving how language models reason over structured information. My research focuses on the intersection of language model reasoning, linguistic structure, and verifiable generation, with particular interest in using formal and executable feedback to evaluate and post-train models. I am also interested in mechanistic interpretability as a means of understanding how these capabilities are represented and emerge internally.

Outside of school, I like to read, watch lots and lots of basketball, and watch movies (and sometimes I write about them which you can find [here](https://letterboxd.com/ayush42/)). When I am not inside I like to play basketball and go bouldering. 

## News

{% assign all_news = site.data.news | sort: 'date' | reverse %}
{% for item in all_news limit: 5 %}
- <span class="small">{{ item.date | date: "%b %d, %Y" }}</span> — {{ item.content | markdownify | remove: '<p>' | remove: '</p>' }}
{% endfor %}