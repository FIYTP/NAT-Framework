# NAT: Neurons Attribution Tracing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A causal intervention framework for mapping the functional roles of individual neurons in Large Language Models (LLMs). NAT identifies three distinct neuronal classes—**Amplifiers**, **Integrators**, and **Non-causal units**—by analyzing network-wide perturbation responses.



## 🧠 Core Idea

Existing interpretability methods often rely on correlational activation analysis or behavioral outputs. NAT shifts the paradigm by **causally intervening on individual neurons** and tracking three signals across the full network:

1.  **Representational Similarity** – Does suppressing this neuron change the model's internal state?
2.  **Compensatory Activation** – Do other neurons increase activity to compensate?
3.  **Inhibitory Response** – Do other neurons decrease activity in response?

These signals naturally separate neurons into three functional roles, revealing a structured computational organization inside LLMs.

## ✨ Key Features

*   **Full-Network Tracking**: Measures perturbations across all layers, not just the target neuron.
*   **Gradio Web Interface**: An interactive UI to configure and run NAT experiments without modifying code.
*   **Cross-Scale Validation**: Tested on Pythia-160M, Pythia-410M, and LLaMA-2-7B.
*   **Functional Portraits & PCA Atlas**: Generate and visualize neuron `portraits` ($F_n$) to map the functional topology of any transformer model.

## 🚀 Quick Start

### 1. Prerequisites

git clone https://github.com/FIYTP/NAT-Framework.git
cd NAT-Framework
pip install -r requirements.txt
2. Prepare Task Data (JSON)
Create a .json file with instruction prompts. For example, you can use the Stanford Alpaca dataset.

json
[
  { "instruction": "Give three tips for staying healthy." },
  { "instruction": "Explain the process of photosynthesis in simple terms." }
]
3. Launch the Interface
bash
python main.py
The Gradio UI will start locally at http://127.0.0.1:7870.



Radar Charts: Per-neuron functional portraits across different task subsets.

Response Heatmaps: Layer-wise compensatory activation and inhibitory response distributions.

🤝 Citation
If you use NAT in your research, please cite our workshop paper:

bibtex
@article{hu2026nat,
  title={NAT: Mapping the Causal Atlas of Functional Neurons Orchestration in LLMs},
  author={Hu, Chenyang},
  journal={ICML 2026 Workshop on Mechanistic Interpretability},
  year={2026},
  note={Under review}
}
📧 Contact
Chenyang Hu – Undergraduate Researcher
Zhejiang Gongshang University, School of Computer Science
Email: [2512190420@pop.zjgsu.edu.cn]
GitHub: @FIYTP
