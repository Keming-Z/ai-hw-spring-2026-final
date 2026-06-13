# Realistic Mini-CLIP Text-to-Image Search

This folder contains a runnable mini research system for text-to-image retrieval using a CLIP-style dual encoder.

## Improvements

1. **Stronger baseline**: compares against a keyword/caption matching method, not only random retrieval.
2. **Harder queries**: evaluation queries use natural phrasing and synonyms like "tiny", "box shape", "three sided shape", and "middle".
3. **Distractor index**: the retrieval index contains many visually similar images that differ by only one attribute, such as color, shape, size, or position.

## How to run

```bash
python main.py
```

The script creates all outputs under `outputs/`:

- `results.json`: metrics from the actual run
- `metrics_chart.png`: Random vs Keyword vs Mini-CLIP chart
- `loss_curve.png`: training loss from the run
- `architecture.png`: system architecture diagram
- `qualitative_results.png`: retrieval examples
- `demo_output.txt`: terminal-style demo output
- `mini_clip_realistic_model.pt`: model checkpoint

## Current run results

The current run used 144 images, 576 training image-text pairs, 48 hard queries, and 18 training epochs.

| Method | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|
| Random expected | 0.7% | 2.1% | 3.5% | 0.039 |
| Keyword baseline | 18.8% | 58.3% | 81.2% | 0.441 |
| Mini-CLIP | 45.8% | 77.1% | 87.5% | 0.634 |

