"""
Neuron Damage & Compensation Analyzer - 独立交互界面
自由添加抑制神经元，实时分析损伤扩散和代偿激活
带完整导出功能 - 修复Dataframe height参数
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
from datetime import datetime
import json
import csv

warnings.filterwarnings('ignore')


# ==================== 模型管理器 ====================
class ModelManager:
    """加载Pythia-160M模型"""

    def __init__(self, model_path: str = r"E:\AI_Models\Pythia\Pythia-160M\models"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.n_layers = 12
        self.d_mlp = 3072
        self._load_model()

    def _load_model(self):
        """加载本地模型"""
        print(f"🚀 加载模型: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.model.eval()
        print(f"✅ 模型加载完成: {self.n_layers}层, {self.d_mlp}神经元/层")

    def get_activations(self,
                        sentence: str,
                        inhibit_neurons: List[Tuple[int, int]] = None) -> np.ndarray:
        """
        获取指定句子的全网激活值
        返回: [n_layers, d_mlp]
        """
        layer_activations = np.zeros((self.n_layers, self.d_mlp))
        captured = {layer: False for layer in range(self.n_layers)}

        # ========== 干预钩子 ==========
        hooks = []

        def create_inhibition_hook(layer_idx, neuron_idx):
            def hook(module, input, output):
                if output is not None:
                    modified = output.clone()
                    modified[..., neuron_idx] = 0.0
                    return modified

            return hook

        if inhibit_neurons:
            for layer, neuron in inhibit_neurons:
                if 0 <= layer < self.n_layers:
                    target_layer = self.model.gpt_neox.layers[layer]
                    hook = target_layer.mlp.act.register_forward_hook(
                        create_inhibition_hook(layer, neuron)
                    )
                    hooks.append(hook)

        # ========== 捕获钩子 ==========
        capture_hooks = []

        def create_capture_hook(layer_idx):
            def hook(module, input, output):
                if output is not None:
                    act = output[0, -1, :].detach().cpu().numpy()
                    layer_activations[layer_idx] = act
                    captured[layer_idx] = True

            return hook

        for layer in range(self.n_layers):
            target_layer = self.model.gpt_neox.layers[layer]
            hook = target_layer.mlp.act.register_forward_hook(
                create_capture_hook(layer)
            )
            capture_hooks.append(hook)

        # ========== 运行推理 ==========
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            _ = self.model(**inputs)

        # ========== 移除钩子 ==========
        for hook in hooks + capture_hooks:
            hook.remove()

        return layer_activations


# ==================== 分析引擎 ====================
class AnalysisEngine:
    """分析损伤扩散和代偿激活"""

    def __init__(self, model_manager: ModelManager):
        self.model = model_manager
        self.n_layers = model_manager.n_layers
        self.d_mlp = model_manager.d_mlp

    def analyze(self,
                sentence: str,
                inhibited_neurons: List[Tuple[int, int]],
                threshold: float = 2.0) -> Dict:
        """
        分析基线和干预的差异
        """
        print(f"\n🔬 分析句子: '{sentence}'")
        print(f"   抑制神经元: {len(inhibited_neurons)}个")

        baseline = self.model.get_activations(sentence, inhibit_neurons=None)
        intervention = self.model.get_activations(sentence, inhibit_neurons=inhibited_neurons)

        inhibition_results = []
        for layer, neuron in inhibited_neurons:
            base_val = baseline[layer, neuron]
            inter_val = intervention[layer, neuron]
            inhibition_results.append({
                'layer': layer,
                'neuron': neuron,
                'baseline': float(base_val),
                'intervention': float(inter_val),
                'success': abs(inter_val) < 1e-6
            })

        diff = intervention - baseline
        std_baseline = np.std(baseline)
        mean_baseline = np.mean(baseline)

        inhibited_mask = np.zeros((self.n_layers, self.d_mlp), dtype=bool)
        for layer, neuron in inhibited_neurons:
            if 0 <= layer < self.n_layers and 0 <= neuron < self.d_mlp:
                inhibited_mask[layer, neuron] = True

        damage_spread = []
        for layer in range(self.n_layers):
            for neuron in range(self.d_mlp):
                if inhibited_mask[layer, neuron]:
                    continue

                drop = baseline[layer, neuron] - intervention[layer, neuron]
                if drop > threshold * std_baseline:
                    damage_spread.append({
                        'layer': layer,
                        'neuron': neuron,
                        'baseline': float(baseline[layer, neuron]),
                        'intervention': float(intervention[layer, neuron]),
                        'drop': float(drop),
                        'drop_percent': float(drop / (abs(baseline[layer, neuron]) + 1e-8) * 100),
                        'z_score': float(drop / std_baseline)
                    })

        compensation = []
        for layer in range(self.n_layers):
            for neuron in range(self.d_mlp):
                if inhibited_mask[layer, neuron]:
                    continue

                rise = intervention[layer, neuron] - baseline[layer, neuron]
                if rise > threshold * std_baseline:
                    compensation.append({
                        'layer': layer,
                        'neuron': neuron,
                        'baseline': float(baseline[layer, neuron]),
                        'intervention': float(intervention[layer, neuron]),
                        'rise': float(rise),
                        'rise_percent': float(rise / (abs(baseline[layer, neuron]) + 1e-8) * 100),
                        'z_score': float(rise / std_baseline)
                    })

        damage_spread.sort(key=lambda x: x['drop'], reverse=True)
        compensation.sort(key=lambda x: x['rise'], reverse=True)

        offsets = []
        for comp in compensation[:100]:
            for inh_layer, _ in inhibited_neurons:
                offsets.append(comp['layer'] - inh_layer)

        offset_counts = {}
        for o in offsets:
            offset_counts[o] = offset_counts.get(o, 0) + 1

        most_common = max(offset_counts.items(), key=lambda x: x[1]) if offset_counts else (0, 0)

        pattern = {
            'inhibited_layers': sorted(list(set([l for l, _ in inhibited_neurons]))),
            'damage_layers': sorted(list(set([d['layer'] for d in damage_spread[:50]]))),
            'compensation_layers': sorted(list(set([c['layer'] for c in compensation[:50]]))),
            'offset_distribution': offset_counts,
            'most_common_offset': most_common[0],
            'downstream_ratio': len([o for o in offsets if o > 0]) / len(offsets) if offsets else 0,
            'same_layer_ratio': len([o for o in offsets if o == 0]) / len(offsets) if offsets else 0,
            'upstream_ratio': len([o for o in offsets if o < 0]) / len(offsets) if offsets else 0,
        }

        return {
            'baseline': baseline,
            'intervention': intervention,
            'inhibition_results': inhibition_results,
            'damage_spread': damage_spread,
            'compensation': compensation,
            'pattern': pattern,
            'statistics': {
                'sentence': sentence,
                'inhibited_count': len(inhibited_neurons),
                'damage_count': len(damage_spread),
                'compensation_count': len(compensation),
                'threshold': threshold,
                'std_baseline': float(std_baseline),
                'mean_baseline': float(mean_baseline)
            }
        }


# ==================== 导出功能 ====================
class DataExporter:
    """处理所有数据导出功能"""

    def __init__(self):
        self.export_dir = Path("neuron_analysis_exports")
        self.export_dir.mkdir(exist_ok=True)

    def export_to_csv(self,
                      damage_df: pd.DataFrame,
                      compensation_df: pd.DataFrame,
                      inhibition_df: pd.DataFrame,
                      sentence: str,
                      inhibited_neurons: List,
                      threshold: float) -> Dict:
        """导出为CSV文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        session_dir = self.export_dir / f"export_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        exported_files = []

        if damage_df is not None and not damage_df.empty:
            damage_path = session_dir / "1_damage_spread.csv"
            damage_df.to_csv(damage_path, index=False, encoding='utf-8-sig')
            exported_files.append(str(damage_path))

        if compensation_df is not None and not compensation_df.empty:
            comp_path = session_dir / "2_compensation.csv"
            compensation_df.to_csv(comp_path, index=False, encoding='utf-8-sig')
            exported_files.append(str(comp_path))

        if inhibition_df is not None and not inhibition_df.empty:
            inhib_path = session_dir / "3_inhibition_verification.csv"
            inhibition_df.to_csv(inhib_path, index=False, encoding='utf-8-sig')
            exported_files.append(str(inhib_path))

        settings = {
            'timestamp': timestamp,
            'sentence': sentence,
            'threshold': threshold,
            'inhibited_neurons': [f"L{l}N{n}" for l, n in inhibited_neurons] if inhibited_neurons else [],
            'inhibited_count': len(inhibited_neurons) if inhibited_neurons else 0,
            'damage_count': len(damage_df) if damage_df is not None else 0,
            'compensation_count': len(compensation_df) if compensation_df is not None else 0
        }

        settings_path = session_dir / "0_experiment_settings.json"
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        exported_files.append(str(settings_path))

        return {
            'success': True,
            'session_dir': str(session_dir),
            'files': exported_files,
            'message': f"✅ 已导出 {len(exported_files)} 个文件到: {session_dir}"
        }

    def export_to_excel(self,
                        damage_df: pd.DataFrame,
                        compensation_df: pd.DataFrame,
                        inhibition_df: pd.DataFrame,
                        sentence: str,
                        inhibited_neurons: List,
                        threshold: float) -> Dict:
        """导出为Excel文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.export_dir / f"export_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        excel_path = session_dir / "neuron_analysis_report.xlsx"

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            settings_df = pd.DataFrame([{
                '实验时间': timestamp,
                '测试句子': sentence,
                '显著阈值': f"{threshold}倍标准差",
                '抑制神经元数量': len(inhibited_neurons) if inhibited_neurons else 0,
                '抑制神经元列表': ', '.join([f"L{l}N{n}" for l, n in inhibited_neurons][:20]) + (
                    '...' if len(inhibited_neurons) > 20 else ''),
                '损伤扩散数量': len(damage_df) if damage_df is not None else 0,
                '代偿激活数量': len(compensation_df) if compensation_df is not None else 0
            }])
            settings_df.to_excel(writer, sheet_name='实验设置', index=False)

            if damage_df is not None and not damage_df.empty:
                damage_df.to_excel(writer, sheet_name='损伤扩散', index=False)

            if compensation_df is not None and not compensation_df.empty:
                compensation_df.to_excel(writer, sheet_name='代偿激活', index=False)

            if inhibition_df is not None and not inhibition_df.empty:
                inhibition_df.to_excel(writer, sheet_name='抑制验证', index=False)

        return {
            'success': True,
            'file': str(excel_path),
            'message': f"✅ 已导出Excel报告: {excel_path}"
        }

    def export_to_json(self, results: Dict) -> Dict:
        """导出完整结果为JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.export_dir / f"export_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        json_path = session_dir / "complete_analysis.json"

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.float32):
                return float(obj)
            if isinstance(obj, np.int64):
                return int(obj)
            return obj

        serializable_results = json.loads(
            json.dumps(results, default=convert_to_serializable)
        )

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        return {
            'success': True,
            'file': str(json_path),
            'message': f"✅ 已导出完整JSON: {json_path}"
        }

    def export_summary_report(self,
                              damage_df: pd.DataFrame,
                              compensation_df: pd.DataFrame,
                              inhibition_df: pd.DataFrame,
                              sentence: str,
                              inhibited_neurons: List,
                              threshold: float,
                              pattern: Dict) -> Dict:
        """导出文本报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.export_dir / f"export_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        report_path = session_dir / "analysis_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🧠 神经元损伤扩散与代偿激活分析报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"📝 测试句子: {sentence}\n")
            f.write(f"📊 显著阈值: {threshold}倍标准差\n")
            f.write(f"🕐 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("-" * 80 + "\n")
            f.write("🎯 抑制神经元列表\n")
            f.write("-" * 80 + "\n")
            for i, (l, n) in enumerate(inhibited_neurons):
                f.write(f"{i + 1:3d}. 层 {l:2d}, 神经元 {n:6d} (L{l}N{n})\n")

            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("📉 损伤扩散统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"总数: {len(damage_df) if damage_df is not None else 0} 个神经元\n\n")

            if damage_df is not None and not damage_df.empty:
                f.write("前20个最显著的损伤扩散神经元:\n")
                f.write(f"{'序号':<6} {'层':<6} {'神经元':<10} {'下降幅度':<12} {'下降比例':<10} {'Z分数':<10}\n")
                f.write("-" * 60 + "\n")
                for i in range(min(20, len(damage_df))):
                    row = damage_df.iloc[i]
                    f.write(
                        f"{i + 1:<6} {row['层']:<6} {row['神经元']:<10} {row['下降幅度']:<12} {row['下降比例']:<10} {row['Z分数']:<10}\n")

            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("📈 代偿激活统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"总数: {len(compensation_df) if compensation_df is not None else 0} 个神经元\n\n")

            if compensation_df is not None and not compensation_df.empty:
                f.write("前20个最显著的代偿激活神经元:\n")
                f.write(f"{'序号':<6} {'层':<6} {'神经元':<10} {'上升幅度':<12} {'上升比例':<10} {'Z分数':<10}\n")
                f.write("-" * 60 + "\n")
                for i in range(min(20, len(compensation_df))):
                    row = compensation_df.iloc[i]
                    f.write(
                        f"{i + 1:<6} {row['层']:<6} {row['神经元']:<10} {row['上升幅度']:<12} {row['上升比例']:<10} {row['Z分数']:<10}\n")

            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("🔍 模式分析\n")
            f.write("-" * 80 + "\n")
            f.write(f"抑制层: {pattern.get('inhibited_layers', [])}\n")
            f.write(f"损伤扩散主要层: {pattern.get('damage_layers', [])[:10]}\n")
            f.write(f"代偿激活主要层: {pattern.get('compensation_layers', [])[:10]}\n")
            f.write(f"\n代偿位置分布:\n")
            f.write(f"  • 下游层: {pattern.get('downstream_ratio', 0):.1%}\n")
            f.write(f"  • 同层:   {pattern.get('same_layer_ratio', 0):.1%}\n")
            f.write(f"  • 上游层: {pattern.get('upstream_ratio', 0):.1%}\n")
            f.write(f"  • 最常见偏移: {pattern.get('most_common_offset', 0)} 层\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("✅ 分析完成\n")
            f.write("=" * 80 + "\n")

        return {
            'success': True,
            'file': str(report_path),
            'message': f"✅ 已导出文本报告: {report_path}"
        }

    def get_export_history(self) -> List:
        """获取导出历史"""
        exports = []
        if self.export_dir.exists():
            for session_dir in sorted(self.export_dir.glob("export_*"), reverse=True)[:10]:
                exports.append(str(session_dir))
        return exports


# ==================== Gradio界面 ====================
class DamageAnalyzerUI:
    """损伤扩散分析交互界面 - 修复Dataframe height参数"""

    def __init__(self):
        print("🧠 初始化神经元损伤分析系统...")
        self.model_manager = ModelManager()
        self.analyzer = AnalysisEngine(self.model_manager)
        self.exporter = DataExporter()
        self.last_results = None
        self.last_damage_df = None
        self.last_comp_df = None
        self.last_inhib_df = None

    def create_interface(self):
        """创建Gradio界面 - 已移除所有height参数"""
        with gr.Blocks(title="🧠 神经元损伤扩散与代偿分析", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🧠 神经元损伤扩散与代偿激活分析")
            gr.Markdown("### 自由添加抑制神经元 → 实时分析 → 一键导出")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"""
                    **模型信息**
                    - 模型: Pythia-160M
                    - 层数: 0-{self.model_manager.n_layers - 1}
                    - 神经元/层: 0-{self.model_manager.d_mlp - 1}
                    """)

            gr.Markdown("---")

            # ========== 左侧：干预设置 ==========
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 🎯 抑制神经元设置")

                    with gr.Group():
                        with gr.Row():
                            inhibit_layer = gr.Number(
                                label="层",
                                minimum=0,
                                maximum=self.model_manager.n_layers - 1,
                                value=5,
                                step=1,
                                precision=0
                            )
                            inhibit_neuron = gr.Number(
                                label="神经元编号",
                                minimum=0,
                                maximum=self.model_manager.d_mlp - 1,
                                value=992,
                                step=1,
                                precision=0
                            )

                    with gr.Row():
                        add_btn = gr.Button("➕ 添加抑制神经元", variant="primary", scale=2)
                        clear_btn = gr.Button("🗑️ 清空列表", variant="secondary", scale=1)

                    # 修复：移除了height参数
                    inhibited_list = gr.Dataframe(
                        label="当前抑制的神经元",
                        headers=["层", "神经元", "ID"],
                        interactive=False,
                        wrap=True
                    )

                    inhibit_count = gr.Textbox(
                        label="抑制数量",
                        value="0 个神经元",
                        interactive=False
                    )

                    gr.Markdown("---")
                    gr.Markdown("## 📝 测试设置")

                    sentence_input = gr.Textbox(
                        label="测试句子",
                        value="Teach me Python.",
                        placeholder="输入测试句子..."
                    )

                    threshold_slider = gr.Slider(
                        label="显著阈值 (标准差倍数)",
                        minimum=1.0,
                        maximum=5.0,
                        value=2.0,
                        step=0.5
                    )

                    analyze_btn = gr.Button("🔬 开始分析", variant="primary", size="lg")

                # ========== 右侧：分析结果 ==========
                with gr.Column(scale=2):
                    gr.Markdown("## 📊 分析结果")

                    with gr.Tabs():
                        # === Tab 1: 损伤扩散列表 ===
                        with gr.TabItem("📉 损伤扩散神经元"):
                            # 修复：移除了height参数
                            damage_df = gr.Dataframe(
                                label="显著低于基线的神经元（被拖累）",
                                headers=["层", "神经元", "基线值", "干预值", "下降幅度", "下降比例", "Z分数"],
                                interactive=False,
                                wrap=True
                            )
                            damage_count = gr.Textbox(label="损伤扩散数量", interactive=False)

                        # === Tab 2: 代偿激活列表 ===
                        with gr.TabItem("📈 代偿激活神经元"):
                            # 修复：移除了height参数
                            compensation_df = gr.Dataframe(
                                label="显著高于基线的神经元（代偿）",
                                headers=["层", "神经元", "基线值", "干预值", "上升幅度", "上升比例", "Z分数"],
                                interactive=False,
                                wrap=True
                            )
                            compensation_count = gr.Textbox(label="代偿激活数量", interactive=False)

                        # === Tab 3: 抑制验证 ===
                        with gr.TabItem("✅ 抑制验证"):
                            # 修复：移除了height参数
                            inhibition_df = gr.Dataframe(
                                label="手动抑制效果验证",
                                headers=["层", "神经元", "基线值", "干预值", "是否成功"],
                                interactive=False,
                                wrap=True
                            )

                        # === Tab 4: 模式分析 ===
                        with gr.TabItem("🔍 模式分析"):
                            with gr.Row():
                                pattern_layers = gr.JSON(label="层分布分析")
                                pattern_offsets = gr.JSON(label="偏移分布分析")

            gr.Markdown("---")

            # ========== 导出功能区 ==========
            gr.Markdown("## 💾 导出数据")

            with gr.Row():
                with gr.Column(scale=1):
                    export_format = gr.Radio(
                        label="导出格式",
                        choices=[
                            "📊 CSV文件 (多个文件)",
                            "📗 Excel文件 (单文件多表格)",
                            "📋 JSON文件 (完整数据)",
                            "📝 文本报告 (可读性)"
                        ],
                        value="📊 CSV文件 (多个文件)"
                    )

                with gr.Column(scale=2):
                    with gr.Row():
                        export_btn = gr.Button("📥 导出当前结果", variant="primary", size="lg")
                        open_folder_btn = gr.Button("📂 打开导出文件夹", variant="secondary")

                    export_status = gr.Textbox(label="导出状态", lines=2)
                    export_files = gr.File(label="下载文件", visible=True, file_count="multiple")

                    with gr.Row():
                        refresh_history_btn = gr.Button("🔄 刷新导出历史", size="sm")
                        export_history = gr.Dropdown(
                            label="最近导出记录",
                            choices=[],
                            interactive=True
                        )

            # ========== 状态管理 ==========
            inhibit_state = gr.State([])

            # ========== 添加抑制神经元 ==========
            def add_inhibited(layer, neuron, current_list):
                current = current_list if current_list else []

                exists = False
                for item in current:
                    if item[0] == layer and item[1] == neuron:
                        exists = True
                        break

                if not exists:
                    current.append([int(layer), int(neuron), f"L{int(layer)}_N{int(neuron)}"])

                if current:
                    df = pd.DataFrame(current, columns=["层", "神经元", "ID"])
                else:
                    df = pd.DataFrame(columns=["层", "神经元", "ID"])

                count_text = f"{len(current)} 个神经元"
                return df, current, count_text

            add_btn.click(
                fn=add_inhibited,
                inputs=[inhibit_layer, inhibit_neuron, inhibit_state],
                outputs=[inhibited_list, inhibit_state, inhibit_count]
            )

            # ========== 清空列表 ==========
            def clear_inhibited():
                df = pd.DataFrame(columns=["层", "神经元", "ID"])
                return df, [], "0 个神经元"

            clear_btn.click(
                fn=clear_inhibited,
                outputs=[inhibited_list, inhibit_state, inhibit_count]
            )

            # ========== 分析函数 ==========
            def run_analysis(sentence, threshold, inhibit_state_data):
                if not sentence.strip():
                    empty_df = pd.DataFrame()
                    return [empty_df] * 3 + ["请输入测试句子"] * 2 + [{}] * 2

                inhibited = []
                if inhibit_state_data:
                    for row in inhibit_state_data:
                        if len(row) >= 2:
                            inhibited.append((int(row[0]), int(row[1])))

                results = self.analyzer.analyze(
                    sentence=sentence,
                    inhibited_neurons=inhibited,
                    threshold=threshold
                )

                self.last_results = results

                damage_data = []
                for d in results['damage_spread'][:500]:
                    damage_data.append([
                        d['layer'],
                        d['neuron'],
                        f"{d['baseline']:.6f}",
                        f"{d['intervention']:.6f}",
                        f"{d['drop']:.6f}",
                        f"{d['drop_percent']:.1f}%",
                        f"{d['z_score']:.2f}"
                    ])

                damage_df = pd.DataFrame(
                    damage_data,
                    columns=["层", "神经元", "基线值", "干预值", "下降幅度", "下降比例", "Z分数"]
                )
                self.last_damage_df = damage_df

                comp_data = []
                for c in results['compensation'][:500]:
                    comp_data.append([
                        c['layer'],
                        c['neuron'],
                        f"{c['baseline']:.6f}",
                        f"{c['intervention']:.6f}",
                        f"{c['rise']:.6f}",
                        f"{c['rise_percent']:.1f}%",
                        f"{c['z_score']:.2f}"
                    ])

                comp_df = pd.DataFrame(
                    comp_data,
                    columns=["层", "神经元", "基线值", "干预值", "上升幅度", "上升比例", "Z分数"]
                )
                self.last_comp_df = comp_df

                inhib_data = []
                for inv in results['inhibition_results']:
                    inhib_data.append([
                        inv['layer'],
                        inv['neuron'],
                        f"{inv['baseline']:.6f}",
                        f"{inv['intervention']:.6f}",
                        "✅" if inv['success'] else "❌"
                    ])

                inhib_df = pd.DataFrame(
                    inhib_data,
                    columns=["层", "神经元", "基线值", "干预值", "是否成功"]
                )
                self.last_inhib_df = inhib_df

                stats = results['statistics']
                damage_text = f"📉 损伤扩散: {stats['damage_count']} 个神经元"
                comp_text = f"📈 代偿激活: {stats['compensation_count']} 个神经元"

                pattern = results['pattern']
                layers_info = {
                    "抑制层": pattern['inhibited_layers'],
                    "损伤扩散主要层": pattern['damage_layers'][:15],
                    "代偿激活主要层": pattern['compensation_layers'][:15],
                }

                offsets_info = {
                    "偏移分布": {str(k): v for k, v in list(pattern['offset_distribution'].items())[:10]},
                    "最常见偏移": f"{pattern['most_common_offset']} 层",
                    "下游比例": f"{pattern['downstream_ratio']:.1%}",
                    "同层比例": f"{pattern['same_layer_ratio']:.1%}",
                    "上游比例": f"{pattern['upstream_ratio']:.1%}"
                }

                return [
                    damage_df, comp_df, inhib_df,
                    damage_text, comp_text,
                    layers_info, offsets_info
                ]

            analyze_btn.click(
                fn=run_analysis,
                inputs=[sentence_input, threshold_slider, inhibit_state],
                outputs=[
                    damage_df, compensation_df, inhibition_df,
                    damage_count, compensation_count,
                    pattern_layers, pattern_offsets
                ]
            )

            # ========== 导出函数 ==========
            def export_results(format_type, sentence, threshold, inhibit_state_data,
                               damage_df, comp_df, inhib_df, pattern):
                inhibited = []
                if inhibit_state_data:
                    inhibited = [(int(row[0]), int(row[1])) for row in inhibit_state_data if len(row) >= 2]

                if "CSV" in format_type:
                    result = self.exporter.export_to_csv(
                        damage_df, comp_df, inhib_df,
                        sentence, inhibited, threshold
                    )
                    import glob
                    files = glob.glob(f"{result['session_dir']}/*.*")
                    return result['message'], files

                elif "Excel" in format_type:
                    result = self.exporter.export_to_excel(
                        damage_df, comp_df, inhib_df,
                        sentence, inhibited, threshold
                    )
                    return result['message'], [result['file']]

                elif "JSON" in format_type:
                    if self.last_results:
                        result = self.exporter.export_to_json(self.last_results)
                        return result['message'], [result['file']]
                    else:
                        return "❌ 没有可导出的数据", []

                elif "文本报告" in format_type:
                    result = self.exporter.export_summary_report(
                        damage_df, comp_df, inhib_df,
                        sentence, inhibited, threshold, pattern
                    )
                    return result['message'], [result['file']]

                return "❌ 未知格式", []

            export_btn.click(
                fn=export_results,
                inputs=[
                    export_format, sentence_input, threshold_slider, inhibit_state,
                    damage_df, compensation_df, inhibition_df, pattern_layers
                ],
                outputs=[export_status, export_files]
            )

            # ========== 打开导出文件夹 ==========
            def open_folder():
                import subprocess
                import platform

                folder_path = str(self.exporter.export_dir.absolute())

                if platform.system() == "Windows":
                    os.startfile(folder_path)
                elif platform.system() == "Darwin":
                    subprocess.run(["open", folder_path])
                else:
                    subprocess.run(["xdg-open", folder_path])

                return f"📂 已打开文件夹: {folder_path}"

            open_folder_btn.click(
                fn=open_folder,
                outputs=[export_status]
            )

            # ========== 刷新导出历史 ==========
            def refresh_export_history():
                exports = self.exporter.get_export_history()
                return gr.Dropdown(choices=exports)

            refresh_history_btn.click(
                fn=refresh_export_history,
                outputs=[export_history]
            )

            # ========== 初始化 ==========
            app.load(
                fn=clear_inhibited,
                outputs=[inhibited_list, inhibit_state, inhibit_count]
            ).then(
                fn=refresh_export_history,
                outputs=[export_history]
            )

        return app


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("🧠 神经元损伤扩散与代偿激活分析系统")
    print("=" * 70)
    print("功能说明:")
    print("  1. 自由添加要抑制的神经元（层，编号）")
    print("  2. 输入测试句子")
    print("  3. 实时分析损伤扩散和代偿激活")
    print("  4. 一键导出多种格式")
    print("=" * 70)

    export_dir = Path("neuron_analysis_exports")
    export_dir.mkdir(exist_ok=True)
    print(f"📁 导出目录: {export_dir.absolute()}")
    print("=" * 70)

    ui = DamageAnalyzerUI()
    app = ui.create_interface()

    print("\n🌐 启动Gradio界面...")
    print("🔗 本地地址: http://127.0.0.1:7865")
    print("📊 分析完成后，点击'导出当前结果'按钮")
    print("📥 支持4种导出格式: CSV, Excel, JSON, 文本报告")
    print("=" * 70)

    app.launch(
        server_name="127.0.0.1",
        server_port=7865,
        share=False,
        quiet=False
    )


if __name__ == "__main__":
    main()