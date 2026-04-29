"""
全脑CT扫描 - Pythia-410M版本
从JSON文件随机选取指定数量的文本，自动与基线对比生成热图
添加随机种子保证可重复性
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import gradio as gr
import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
from datetime import datetime
import json
import random

warnings.filterwarnings('ignore')


# ==================== 设置随机种子 ====================
def set_seed(seed=42):
    """设置所有随机种子，保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🎲 随机种子已设置为: {seed}")


# 全局设置种子
set_seed(42)

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    "name": "Pythia-410M",
    "path": r"E:\AI_Models\Pythia\Pythia-410M\models",
    "layers": 24,
    "neurons_per_layer": 4096,
    "port": 7861
}


# ==================== JSON提示读取器 ====================
class PromptReader:
    """从JSON文件读取提示"""

    @staticmethod
    def read_prompts(json_path: str) -> List[str]:
        """读取JSON文件中的所有提示"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        prompts = []
        for item in data:
            if isinstance(item, dict) and 'instruction' in item:
                prompts.append(item['instruction'])
            elif isinstance(item, str):
                prompts.append(item)

        print(f"📖 从 {json_path} 读取了 {len(prompts)} 个提示")
        return prompts

    @staticmethod
    def random_select(prompts: List[str], num: int) -> List[str]:
        """
        随机选择指定数量的提示
        由于已设置全局种子，每次运行结果相同
        """
        if num >= len(prompts):
            return prompts.copy()
        return random.sample(prompts, num)


# ==================== 激活收集器 ====================
class ActivationCollector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.n_layers = MODEL_CONFIG["layers"]
        self.d_mlp = MODEL_CONFIG["neurons_per_layer"]
        self._load_model()

    def _load_model(self):
        """加载模型"""
        print(f"Loading {MODEL_CONFIG['name']} from: {self.model_path}")

        # 先加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 检查是否有GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True
        ).to(device)

        self.model.eval()
        print(f"✅ {MODEL_CONFIG['name']} loaded: {self.n_layers} layers, {self.d_mlp} neurons/layer")

    def _setup_hooks(self):
        """设置钩子收集FFN激活后值"""
        self.activations_sum = {}
        self.activations_count = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                if output is not None:
                    act = output[0, -1, :].detach().cpu()
                    key = f"layer_{layer_idx}"
                    if key not in self.activations_sum:
                        self.activations_sum[key] = torch.zeros_like(act)
                        self.activations_count[key] = 0
                    self.activations_sum[key] += act
                    self.activations_count[key] += 1

            return hook

        self.hooks.clear()
        for idx, layer in enumerate(self.model.gpt_neox.layers):
            hook = layer.mlp.act.register_forward_hook(make_hook(idx))
            self.hooks.append(hook)

    def _cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def collect_average_activations(self, text: str, num_iterations: int = 10) -> Dict:
        """收集单个文本的平均激活值"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        self.activations_sum = {}
        self.activations_count = {}
        self._setup_hooks()

        with torch.no_grad():
            for i in range(num_iterations):
                _ = self.model(**inputs)

        avg_activations = {}
        for layer_key in self.activations_sum:
            sum_act = self.activations_sum[layer_key]
            count = self.activations_count[layer_key]
            if count > 0:
                avg_activations[layer_key] = (sum_act / count).numpy()
            else:
                avg_activations[layer_key] = np.zeros(self.d_mlp)

        self._cleanup()
        return avg_activations

    def collect_multiple_texts(self, texts: List[str], num_iterations: int = 10) -> Dict:
        """收集多个文本的平均激活值（按层平均）"""
        # 初始化累积器
        all_layers = set()
        first_text_acts = self.collect_average_activations(texts[0], num_iterations)
        for layer_key in first_text_acts.keys():
            all_layers.add(layer_key)

        # 初始化累加数组
        sum_activations = {layer: np.zeros_like(first_text_acts[layer]) for layer in all_layers}
        count = 0

        for text in texts:
            print(f"   Processing: {text[:50]}...")
            acts = self.collect_average_activations(text, num_iterations)
            for layer, values in acts.items():
                sum_activations[layer] += values
            count += 1

        # 计算平均值
        avg_activations = {}
        for layer, values in sum_activations.items():
            avg_activations[layer] = values / count

        return avg_activations

    def save_to_csv(self, activations: Dict, description: str,
                    custom_filename: str = "", output_dir: str = "results_410m") -> str:
        """保存激活值到CSV"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if custom_filename:
            filename = f"410m_{custom_filename}.csv"
        else:
            filename = f"410m_activations_{description}_{timestamp}.csv"

        filepath = output_path / filename

        rows = []
        for layer_key, values in activations.items():
            layer_idx = int(layer_key.split("_")[1])
            for neuron_idx, value in enumerate(values):
                rows.append({
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "average_activation": float(value)
                })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

        # 保存元数据
        meta = {
            "model": MODEL_CONFIG["name"],
            "seed": 42,
            "description": description,
            "timestamp": timestamp,
            "total_neurons": len(rows),
            "total_layers": len(activations),
            "file_path": str(filepath)
        }
        meta_df = pd.DataFrame([meta])
        meta_filename = f"{filepath.stem}_meta.csv"
        meta_df.to_csv(output_path / meta_filename, index=False)

        print(f"✓ Data saved to: {filepath}")
        return str(filepath)


# ==================== 增强版对比分析器 ====================
class EnhancedActivationComparator:
    @staticmethod
    def compare_with_baseline(baseline_file: str, target_file: str,
                              output_dir: str = "comparison_results_410m") -> Tuple[pd.DataFrame, plt.Figure, str, str]:
        """
        将目标文件与基线文件对比，生成热图和详细CSV
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 加载数据
        df_base = pd.read_csv(baseline_file)
        df_target = pd.read_csv(target_file)

        # 合并数据
        merged = pd.merge(
            df_base, df_target,
            on=["layer", "neuron"],
            suffixes=("_baseline", "_target"),
            how="inner"
        )

        if merged.empty:
            raise ValueError("No common neurons found between files")

        # 计算差异
        merged["difference"] = abs(merged["average_activation_target"] - merged["average_activation_baseline"])

        # 找出差异最大的神经元（跨层汇总）
        neuron_total_diffs = merged.groupby("neuron")["difference"].sum().sort_values(ascending=False)
        top_neurons = neuron_total_diffs.head(40).index.tolist()

        # 获取所有层
        all_layers = sorted(merged["layer"].unique())
        total_layers = len(all_layers)

        # 构建热图矩阵
        heatmap_matrix = np.zeros((len(top_neurons), total_layers))
        for i, neuron in enumerate(top_neurons):
            neuron_data = merged[merged["neuron"] == neuron]
            for _, row in neuron_data.iterrows():
                layer_idx = int(row["layer"])
                col_idx = all_layers.index(layer_idx)
                heatmap_matrix[i, col_idx] = float(row["difference"])

        # 保存详细数据
        detailed_data = []
        for neuron in top_neurons:
            neuron_data = merged[merged["neuron"] == neuron]
            for _, row in neuron_data.iterrows():
                detailed_data.append({
                    "layer": int(row["layer"]),
                    "neuron": int(row["neuron"]),
                    "baseline": float(row["average_activation_baseline"]),
                    "target": float(row["average_activation_target"]),
                    "difference": float(row["difference"]),
                    "percent_change": float(
                        (row["difference"] / (abs(row["average_activation_baseline"]) + 1e-8)) * 100)
                })

        detailed_df = pd.DataFrame(detailed_data)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(baseline_file).replace(".csv", "")
        target_name = os.path.basename(target_file).replace(".csv", "")

        # 保存详细CSV
        csv_filename = f"410m_comparison_{target_name}_vs_{base_name}_{timestamp}.csv"
        csv_path = output_path / csv_filename
        detailed_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 创建热图
        fig = EnhancedActivationComparator._create_heatmap(
            heatmap_matrix, top_neurons, all_layers,
            base_name, target_name, merged
        )

        # 保存热图
        heatmap_filename = f"410m_heatmap_{target_name}_vs_{base_name}_{timestamp}.png"
        heatmap_path = output_path / heatmap_filename
        fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')

        # 返回每层差异最大的神经元
        top_diffs_by_layer = []
        for layer in all_layers:
            layer_data = merged[merged["layer"] == layer]
            top_in_layer = layer_data.nlargest(10, "difference")
            top_diffs_by_layer.append(top_in_layer)

        result_df = pd.concat(top_diffs_by_layer) if top_diffs_by_layer else pd.DataFrame()
        result_df = result_df.sort_values(["layer", "difference"], ascending=[True, False])

        return result_df, fig, str(csv_path), str(heatmap_path)

    @staticmethod
    def _create_heatmap(matrix, top_neurons, all_layers, base_name, target_name, merged):
        """创建热图"""
        num_neurons, num_layers = matrix.shape

        # 计算颜色范围
        all_values = matrix[matrix > 0]
        if len(all_values) > 0:
            vmin = 0
            vmax = np.percentile(all_values, 98)
            if vmax <= 0:
                vmax = np.max(all_values) if len(all_values) > 0 else 1.0
        else:
            vmin, vmax = 0, 1.0

        # 创建图表
        fig_width = max(12, num_layers * 0.8)
        fig_height = max(8, num_neurons * 0.3)

        fig, (ax, ax_cbar) = plt.subplots(1, 2, figsize=(fig_width + 2, fig_height),
                                          gridspec_kw={'width_ratios': [fig_width, 0.3]})

        im = ax.imshow(matrix, aspect='auto', cmap='viridis',
                       interpolation='nearest', vmin=vmin, vmax=vmax)

        # 设置标签
        ax.set_xticks(np.arange(num_layers))
        ax.set_yticks(np.arange(num_neurons))
        ax.set_xticklabels([f"L{layer}" for layer in all_layers], fontsize=8, rotation=45)
        ax.set_yticklabels([f"N{neuron}" for neuron in top_neurons], fontsize=7)

        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Neuron ID", fontsize=10)

        ax.set_title(f"Pythia-410M: Activation Differences (Seed=42)\n{target_name} vs {base_name}",
                     fontsize=12, fontweight='bold')

        # 颜色条
        cbar = plt.colorbar(im, cax=ax_cbar)
        cbar.set_label('Difference', fontsize=9)

        plt.tight_layout()
        return fig


# ==================== Gradio界面 ====================
class GradioApp:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.collector = None
        self.all_prompts = []
        self.baseline_file = None

    def load_json_prompts(self, json_file):
        if json_file is None:
            self.all_prompts = []
            return "No JSON file uploaded"
        try:
            self.all_prompts = PromptReader.read_prompts(json_file.name)
            return f"✅ Loaded {len(self.all_prompts)} prompts (Seed=42)"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def set_baseline(self, baseline_file):
        """设置基线文件"""
        if baseline_file is None:
            self.baseline_file = None
            return "No baseline file selected"
        self.baseline_file = baseline_file.name
        return f"✅ Baseline set: {os.path.basename(baseline_file.name)}"

    def run_test(self, json_file, num_texts, num_iterations, progress=gr.Progress()):
        """运行测试"""
        try:
            # 再次确认种子（防止外部干扰）
            set_seed(42)

            if not self.all_prompts:
                return "Please load JSON file first", None, None, None, None

            if self.baseline_file is None:
                return "Please select baseline CSV file first", None, None, None, None

            # 1. 随机选择文本
            progress(0.1, desc="Randomly selecting texts...")
            selected_prompts = PromptReader.random_select(self.all_prompts, num_texts)

            # 2. 初始化收集器
            progress(0.2, desc="Initializing model...")
            if self.collector is None:
                self.collector = ActivationCollector(self.model_path)

            # 3. 收集激活值
            progress(0.4, desc=f"Collecting activations for {len(selected_prompts)} texts...")
            description = f"random_{num_texts}_texts"
            avg_activations = self.collector.collect_multiple_texts(selected_prompts, num_iterations)

            # 4. 保存CSV
            progress(0.7, desc="Saving results...")
            target_file = self.collector.save_to_csv(
                avg_activations,
                description,
                custom_filename=f"random_{num_texts}_texts"
            )

            # 5. 与基线对比
            progress(0.8, desc="Comparing with baseline...")
            result_df, heatmap_fig, csv_path, heatmap_path = EnhancedActivationComparator.compare_with_baseline(
                self.baseline_file, target_file
            )

            # 6. 准备结果信息
            status = f"""
            ✅ Test completed! (Seed=42)

            📝 Selected {len(selected_prompts)} random texts:
            """
            for i, p in enumerate(selected_prompts[:5]):
                status += f"\n   {i + 1}. {p[:80]}..."
            if len(selected_prompts) > 5:
                status += f"\n   ... and {len(selected_prompts) - 5} more"

            status += f"""

            📊 Results saved:
            - Target CSV: {os.path.basename(target_file)}
            - Comparison CSV: {os.path.basename(csv_path)}
            - Heatmap: {os.path.basename(heatmap_path)}

            🔬 All random processes used seed=42 for reproducibility
            """

            progress(1.0, desc="Complete")

            return status, heatmap_fig, csv_path, heatmap_path, target_file

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", None, None, None, None

    def create_interface(self):
        with gr.Blocks(title=f"{MODEL_CONFIG['name']} - Random Text Test", theme=gr.themes.Soft()) as app:
            gr.Markdown(f"# 🧠 {MODEL_CONFIG['name']} - Random Text Test")
            gr.Markdown(f"### {MODEL_CONFIG['layers']} Layers | {MODEL_CONFIG['neurons_per_layer']} Neurons/Layer")
            gr.Markdown("### 🎲 Fixed Random Seed: 42 (Reproducible Results)")

            with gr.Row():
                with gr.Column(scale=1):
                    # JSON文件上传
                    json_file = gr.File(
                        label="📁 Upload JSON prompts file",
                        file_types=[".json"]
                    )
                    json_status = gr.Textbox(label="JSON Status", value="No file", interactive=False)

                    # 基线文件选择
                    baseline_file = gr.File(
                        label="📊 Select baseline CSV file",
                        file_types=[".csv"]
                    )
                    baseline_status = gr.Textbox(label="Baseline Status", value="No file", interactive=False)

                    # 测试参数
                    num_texts = gr.Slider(
                        label="Number of random texts",
                        minimum=1,
                        maximum=200,
                        value=10,
                        step=1
                    )

                    num_iterations = gr.Slider(
                        label="Iterations per text",
                        minimum=1,
                        maximum=50,
                        value=5,
                        step=1
                    )

                    run_btn = gr.Button("🚀 Run Random Test", variant="primary", size="lg")

                    status_text = gr.Textbox(label="Status", lines=12, interactive=False)

                with gr.Column(scale=2):
                    heatmap_plot = gr.Plot(label="Difference Heatmap")

                    with gr.Row():
                        comparison_csv = gr.File(label="📥 Download Comparison CSV", visible=True)
                        heatmap_file = gr.File(label="📥 Download Heatmap PNG", visible=True)
                        target_csv = gr.File(label="📥 Download Target CSV", visible=True)

            # 事件绑定
            json_file.change(
                fn=self.load_json_prompts,
                inputs=[json_file],
                outputs=[json_status]
            )

            baseline_file.change(
                fn=self.set_baseline,
                inputs=[baseline_file],
                outputs=[baseline_status]
            )

            run_btn.click(
                fn=self.run_test,
                inputs=[json_file, num_texts, num_iterations],
                outputs=[status_text, heatmap_plot, comparison_csv, heatmap_file, target_csv]
            )

        return app


# ==================== 主程序 ====================
def main():
    if not os.path.exists(MODEL_CONFIG["path"]):
        print(f"⚠️ Warning: Model path does not exist: {MODEL_CONFIG['path']}")
        return

    print("=" * 70)
    print(f"🚀 Starting {MODEL_CONFIG['name']} - Random Text Test")
    print("=" * 70)
    print("Features:")
    print(f"  • Model: {MODEL_CONFIG['name']}")
    print(f"  • Layers: {MODEL_CONFIG['layers']}")
    print(f"  • Neurons/layer: {MODEL_CONFIG['neurons_per_layer']}")
    print("  1. Load prompts from JSON file")
    print("  2. Select baseline CSV file")
    print("  3. Randomly select N texts")
    print("  4. Auto-generate heatmap and comparison CSV")
    print(f"  5. 🎲 Fixed random seed: 42 (reproducible results)")
    print("=" * 70)

    app = GradioApp(MODEL_CONFIG["path"])
    interface = app.create_interface()

    print(f"\n🌐 Gradio interface starting...")
    print(f"👉 Local access: http://127.0.0.1:{MODEL_CONFIG['port']}")
    print("=" * 70)

    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=MODEL_CONFIG["port"],
            share=False,
            quiet=False
        )
    except Exception as e:
        print(f"❌ Launch failed: {e}")


if __name__ == "__main__":
    main()