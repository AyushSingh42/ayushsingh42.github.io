---
layout: default
title: Research
permalink: /research/
---

# Research

I am affiliated with the UIUC Computational Linguistics Laboratory, specfically the Computational Semantics Lab. Currently, I am working with Professor Aleksandre Maskharashvili on natural language generation from table statistics to text using generalized quantifiers.

I am also a SPAR AI Fellow where I am working on exploring sparse representations in LLMs using SAEs, LoRA, latent geometry analysis, and formal verification tools, under Dr. Yuxiao Li. 

Previously, I conducted research in developmental psycholinguistics at Rutgers University through the Aresty Summer Science Program. Working in the Laboratory for Developmental Language Studies, I investigated how children learn and classify novel words related to mental and emotional states. I presented this work as a [poster](/assets/papers/aresty-poster.pdf) titled *Learning Emotion and Mental State Adjectives from Linguistic Context* in August 2024.


{% for r in site.research %}
- [{{ r.title }}]({{ r.url | relative_url }})
{% endfor %}
