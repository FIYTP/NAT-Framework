# NAT: Neurons Attribution Tracing

A causal intervention framework for analyzing what individual neurons actually *do* inside large language models.

## What this is

When you poke a neuron in an LLM and it doesn't break anything, is that neuron useless? Not necessarily — the network might just be rerouting around it. When you poke another neuron and the whole thing collapses, why?

NAT is a framework I built to answer these kinds of questions. It doesn't just look at whether output changes — it watches how the entire network reorganizes itself after each intervention.

## How it works

1. Find neurons that light up for structured tasks (step-by-step instructions, formatting, etc.)
2. Suppress them one at a time
3. Measure three things:
   - Did the internal representation actually change?
   - Did other neurons spike to compensate?
   - Did other neurons go quiet?
4. Based on these signals, each neuron falls into one of three categories:
   - **Amplifiers**: suppress them, the network compensates like crazy, output stays fine
   - **Integrators**: suppress them alongside amplifiers, everything falls apart
   - **Non-causal**: they react to everything but do nothing when removed

Tested on Pythia-160M, Pythia-410M, and LLaMA-2-7B. The pattern holds across scales, but the redundancy shrinks as models get bigger.

## Quick start

```bash
git clone https://github.com/FIYTP/NAT-Framework.git
cd NAT-Framework
pip install -r requirements.txt
python main.py
```

The Gradio interface opens at `http://127.0.0.1:7870`. You'll need a local copy of LLaMA or Pythia to run experiments.

## Files

```
NAT-Framework/
├── main.py           # Launch the Gradio UI
├── nat-engine/
│   ├── model.py      # Model loading, hooks, suppression
│   ├── engine.py     # Core experiment logic
│   └── viz.py        # Atlas, radar plots, heatmaps
├── data/             # Sample task prompts (JSON)
├── results/          # Output CSVs and figures
└── requirements.txt
```

## Requirements

- PyTorch, Transformers, BitsAndBytes
- Gradio (for the UI)
- scikit-learn, matplotlib, pandas
- A GPU with enough VRAM for the model you're testing

## Contact

Chenyang Hu
Zhejiang Gongshang University, School of Computer Science
GitHub: [@FIYTP](https://github.com/FIYTP)
