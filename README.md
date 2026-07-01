# Siyu Chen Blog

Personal AI research notes and engineering blog built with [Hexo](https://hexo.io/) and [Fluid](https://github.com/fluid-dev/hexo-theme-fluid).

## Local Development

```bash
npm install
npm run server
```

## Writing

```bash
npm run new:post "paper-title-or-topic"
```

Useful front matter fields:

```yaml
---
title: Paper Note Title
date: 2026-07-01 22:00:00
updated: 2026-07-01 22:00:00
categories:
  - Paper Notes
tags:
  - LLM
  - Alignment
math: true
mermaid: true
---
```

The site is deployed from `main` to the `gh-pages` branch by GitHub Actions.

