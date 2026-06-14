---
layout: default
title: Home
---

<div class="page">
  <section class="hero">
    <p class="overline">Personal Blog · Machine Learning · Engineering</p>
    <h1>THE AGENT THAT<br>GROWS WITH YOU.</h1>
    <p>Notes on machine learning, data systems, and the occasional analog curiosity. I write about things I build and things I learn.</p>
  </section>

  <section>
    <p class="section-label">Latest Posts</p>
    <div class="posts-grid">
      {% for post in site.posts %}
      <article class="post-row">
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        <time datetime="{{ post.date | date: '%Y-%m-%d' }}">{{ post.date | date: '%B %d, %Y' }}</time>
      </article>
      {% endfor %}
    </div>
  </section>

  <section>
    <p class="section-label">What to expect</p>
    <div class="feature-grid">
      <div class="feature-cell">
        <h3>Systems</h3>
        <p>Memory, throughput, and tooling that let models fit in real hardware constraints.</p>
      </div>
      <div class="feature-cell">
        <h3>Models</h3>
        <p>From training pipelines to unconventional architectures and tiny edge inference.</p>
      </div>
      <div class="feature-cell">
        <h3>Experiments</h3>
        <p>Small PoCs that connect machine learning to physics, silicon, and curiosity.</p>
      </div>
    </div>
  </section>

  <section class="terminal">
    <div class="terminal-header">
      <div class="terminal-dots">
        <span class="dot red" aria-hidden="true"></span>
        <span class="dot yellow" aria-hidden="true"></span>
        <span class="dot green" aria-hidden="true"></span>
      </div>
      <span class="terminal-title">Terminal</span>
    </div>
    <div class="terminal-body">
      <code>$ git clone https://github.com/habrzyk-pawel/blog.git</code><br>
      <code>$ cd blog && jekyll serve</code><br>
      <span style="color: var(--ink-soft);"># open http://localhost:4000</span>
    </div>
  </section>
</div>
