---
title: Home
layout: home
---

Welcome to my blog.

This site is now set up to publish posts written in Markdown. Add a new file under the _posts folder with the format `YYYY-MM-DD-your-post-title.md` and GitHub Pages will turn it into a post automatically.

## Recent posts

{% raw %}
<ul>
{% for post in site.posts %}
	<li><a href="{{ post.url }}">{{ post.title }}</a> — <small>{{ post.date | date: "%Y-%m-%d" }}</small></li>
{% endfor %}
</ul>
{% endraw %}

If your new post doesn't appear immediately on GitHub Pages, ensure the file is committed and pushed to the repository's default branch.
