# Habrzyk-Pawel.github.io

Personal blog powered by Jekyll and Hermes Agent.

## About

This is my technical blog covering topics like machine learning, systems engineering, and AI research. The design is inspired by the [Hermes Agent](https://hermes-agent.nousresearch.com/) aesthetic — clean, engineering-focused, and minimal.

## Tech Stack

- **Static Site Generator:** Jekyll
- **Agent Framework:** Hermes Agent (Nous Research)
- **Hosting:** GitHub Pages
- **Model Provider:** Parasail API (custom endpoint)

## Available Models

The following models are available through the Parasail provider:

| Provider | Model ID | Best For |
|----------|----------|----------|
| Moonshot | `parasail-kimi-k27-code` | Coding, complex reasoning, multi-step tasks |
| Qwen | `parasail-qwen3-coder-next` | Code generation, technical writing |
| Meta | `parasail-llama-33-70b-fp8` | Instruction-following, stable agent behavior |
| Meta | `parasail-llama-4-maverick-instruct-fp8` | Multimodal, modern architecture |
| Google | `parasail-gemma-4-31b-it` | Balance of speed and reasoning |
| Google | `parasail-gemma-4-26b-a4b-it` | Efficient general-purpose tasks |
| DeepSeek | `parasail-deepseek-v4-flash` | Large context, fast processing |
| DeepSeek | `parasail-deepseek-v4-pro` | Advanced reasoning, math |
| MiniMax | `parasail-minimax-m3` | Long documents, technical docs |
| UI-TARS | `parasail-ui-tars-1p5-7b` | Visual tasks, UI interaction |
| Cydonia | `parasail-cydonia-24-v41` | Creative writing, roleplay |

## Features

- **Hermes Agent Integration:** Uses Hermes for automated blog post creation and content generation.
- **LaTeX Support:** Mathematical formulas rendered with HTML entities.
- **Syntax Highlighting:** Code snippets with visual styling.
- **Responsive Design:** Mobile-friendly layout with grid-based navigation.

## Local Development

```bash
# Install dependencies
bundle install

# Serve locally
bundle exec jekyll serve

# Build
bundle exec jekyll build
```

## License

MIT License. See `LICENSE` file for details.

## Contact

- GitHub: [@habrzyk-pawel](https://github.com/habrzyk-pawel)
- Blog: [habrzyk-pawel.github.io](https://habrzyk-pawel.github.io)
