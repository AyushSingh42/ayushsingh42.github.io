---
layout: default
title: Blog
permalink: /blog/
---

# Blog

{% for post in site.posts %}
<article class="blog-entry">
  <h2 class="blog-title">
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </h2>

  <div class="blog-date">
    {{ post.date | date: "%B %d, %Y" }}
  </div>

  {% if post.description %}
    <p class="blog-desc">{{ post.description }}</p>
  {% else %}
    <p class="blog-desc">{{ post.excerpt | strip_html | truncate: 220 }}</p>
  {% endif %}
</article>
{% endfor %}
