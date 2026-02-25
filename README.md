# Thematic-LM: A LLM-based Multi-agent System for Large-scale Thematic Analysis

An open-source implementation of the Thematic-LM paper using the [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk).

## Overview

Thematic-LM is a multi-agent system for large-scale computational thematic analysis. It addresses key challenges in analyzing large datasets by:

- **Distributing tasks among specialized agents**: Coder, Aggregator, Reviewer, and Theme Coder agents work together
- **Maintaining an adaptive codebook**: Codes are refined and updated as new data is processed
- **Assigning different identity perspectives**: Diverse viewpoints lead to richer thematic analysis

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CODING STAGE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                               │
│   │ Coder 1 │     │ Coder 2 │     │ Coder N │                               │
│   │(Identity)│     │(Identity)│     │(Identity)│                               │
│   └────┬────┘     └────┬────┘     └────┬────┘                               │
│        │               │               │                                     │
│        └───────────────┼───────────────┘                                     │
│                        ▼                                                     │
│                 ┌──────────────┐                                             │
│                 │  Aggregator  │                                             │
│                 └──────┬───────┘                                             │
│                        │                                                     │
│                        ▼                                                     │
│                 ┌──────────────┐     ┌──────────────┐                        │
│                 │   Reviewer   │◄───►│   Codebook   │                        │
│                 └──────────────┘     │ (Embeddings) │                        │
│                                      └──────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      THEME DEVELOPMENT STAGE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                       │
│   │Theme Coder 1│   │Theme Coder 2│   │Theme Coder N│                       │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                       │
│          │                 │                 │                               │
│          └─────────────────┼─────────────────┘                               │
│                            ▼                                                 │
│                   ┌────────────────┐                                         │
│                   │Theme Aggregator│                                         │
│                   └────────┬───────┘                                         │
│                            │                                                 │
│                            ▼                                                 │
│                    ┌──────────────┐                                          │
│                    │ Final Themes │                                          │
│                    └──────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

## Quick Start

```python
from thematic_lm import ThematicLMPipeline
from thematic_lm.config import ThematicLMConfig

# Configure the pipeline
config = ThematicLMConfig(
    num_coders=2,
    num_theme_coders=2,
    top_k_similar=10,
    max_quotes_per_code=20,
)

# Create and run the pipeline
pipeline = ThematicLMPipeline(config)
themes = pipeline.analyze(data)

print(themes)
```

## Features

- **Multi-agent system**: Specialized agents for coding, aggregation, review, and theme development
- **Adaptive codebook**: Maintains and updates codes using embedding similarity
- **Identity perspectives**: Simulate diverse coder viewpoints (progressive, conservative, etc.)
- **Evaluation framework**: Assess credibility, dependability, and transferability of themes

## Evaluation Metrics

Based on trustworthiness principles in qualitative research:

1. **Credibility & Confirmability**: Measures theme-data consistency using LLM-as-judge
2. **Dependability**: Measures stability via inter-rater reliability (ROUGE scores)
3. **Transferability**: Measures generalization across dataset splits

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{qiao2025thematiclm,
  title={Thematic-LM: A LLM-based Multi-agent System for Large-scale Thematic Analysis},
  author={Qiao, Tingrui and Walker, Caroline and Cunningham, Chris and Koh, Yun Sing},
  booktitle={Proceedings of the ACM Web Conference 2025 (WWW '25)},
  year={2025},
  doi={10.1145/3696410.3714595}
}
```

## Acknowledgments

This implementation is based on the Thematic-LM paper and built using the [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk).
