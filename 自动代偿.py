"""
自动化神经元干预测试 - 网页交互界面
上传CSV文件，自动测试所有显著差异神经元
输出可下载的CSV文件
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import pandas as pd
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
from datetime import datetime
import json
import gc
import time
import traceback
import tempfile

warnings.filterwarnings('ignore')


# ==================== 内存管理工具 ====================
def cleanup_memory():
    """强制清理GPU和CPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ==================== 模型管理器 ====================
class ModelManager:
    """加载Llama2 7B 4bit模型"""

    def __init__(self, model_path: str = r"E:\AI_Models\Llama2 7B\models"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.n_layers = 32
        self.d_mlp = 11008
        self._active_hooks = []
        self._load_model()

    def _load_model(self):
        """加载模型"""
        print("🦙 Loading Llama2 7B 4bit model...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )

        self.model.eval()
        print(f"✅ Model loaded: {self.n_layers} layers, {self.d_mlp} neurons/layer")

    def clear_hooks(self):
        """清除所有活动的钩子"""
        for hook in self._active_hooks:
            try:
                hook.remove()
            except:
                pass
        self._active_hooks.clear()
        cleanup_memory()

    def get_baseline_activations(self, sentence: str) -> np.ndarray:
        """获取基线激活值（无抑制）"""
        self.clear_hooks()

        layer_activations = np.zeros((self.n_layers, self.d_mlp))

        def create_capture_hook(layer_idx):
            def hook(module, input, output):
                if output is not None:
                    if isinstance(output, tuple):
                        act = output[0][0, -1, :].detach().cpu().float().numpy()
                    else:
                        act = output[0, -1, :].detach().cpu().float().numpy()
                    layer_activations[layer_idx] = act
            return hook

        # 注册捕获钩子
        for layer in range(self.n_layers):
            target_layer = self.model.model.layers[layer]
            if hasattr(target_layer, 'mlp'):
                mlp = target_layer.mlp
                if hasattr(mlp, 'act_fn'):
                    hook = mlp.act_fn.register_forward_hook(create_capture_hook(layer))
                    self._active_hooks.append(hook)
                elif hasattr(mlp, 'gate_proj'):
                    hook = mlp.gate_proj.register_forward_hook(create_capture_hook(layer))
                    self._active_hooks.append(hook)

        # 运行推理
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            _ = self.model(**inputs)

        self.clear_hooks()
        return layer_activations

    def get_intervention_activations(self, sentence: str, inhibit_layer: int, inhibit_neuron: int) -> np.ndarray:
        """获取抑制特定神经元后的激活值"""
        self.clear_hooks()

        layer_activations = np.zeros((self.n_layers, self.d_mlp))

        # 抑制钩子
        def create_inhibition_hook():
            def hook(module, input, output):
                if output is not None:
                    if isinstance(output, tuple):
                        modified = output[0].clone()
                        if inhibit_neuron < modified.shape[-1]:
                            modified[..., inhibit_neuron] = 0.0
                        return (modified,) + output[1:]
                    else:
                        modified = output.clone()
                        if inhibit_neuron < modified.shape[-1]:
                            modified[..., inhibit_neuron] = 0.0
                        return modified
                return output
            return hook

        # 注册抑制钩子
        if 0 <= inhibit_layer < self.n_layers:
            target_layer = self.model.model.layers[inhibit_layer]
            if hasattr(target_layer, 'mlp'):
                mlp = target_layer.mlp
                if hasattr(mlp, 'act_fn'):
                    hook = mlp.act_fn.register_forward_hook(create_inhibition_hook())
                    self._active_hooks.append(hook)
                elif hasattr(mlp, 'gate_proj'):
                    hook = mlp.gate_proj.register_forward_hook(create_inhibition_hook())
                    self._active_hooks.append(hook)

        # 捕获钩子
        def create_capture_hook(layer_idx):
            def hook(module, input, output):
                if output is not None:
                    if isinstance(output, tuple):
                        act = output[0][0, -1, :].detach().cpu().float().numpy()
                    else:
                        act = output[0, -1, :].detach().cpu().float().numpy()
                    layer_activations[layer_idx] = act
            return hook

        for layer in range(self.n_layers):
            target_layer = self.model.model.layers[layer]
            if hasattr(target_layer, 'mlp'):
                mlp = target_layer.mlp
                if hasattr(mlp, 'act_fn'):
                    hook = mlp.act_fn.register_forward_hook(create_capture_hook(layer))
                    self._active_hooks.append(hook)
                elif hasattr(mlp, 'gate_proj'):
                    hook = mlp.gate_proj.register_forward_hook(create_capture_hook(layer))
                    self._active_hooks.append(hook)

        # 运行推理
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            _ = self.model(**inputs)

        self.clear_hooks()
        return layer_activations


# ==================== 分析引擎 ====================
class AnalysisEngine:
    """分析损伤扩散和代偿激活"""

    def __init__(self):
        pass

    def analyze(self,
                baseline: np.ndarray,
                intervention: np.ndarray,
                inhibited_neurons: List[Tuple[int, int]],
                threshold: float = 2.0) -> Dict:
        """
        分析基线和干预的差异
        """
        n_layers, d_mlp = baseline.shape

        # 验证抑制效果
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

        std_baseline = np.std(baseline)
        mean_baseline = np.mean(baseline)

        # 标记手动抑制的神经元
        inhibited_mask = np.zeros((n_layers, d_mlp), dtype=bool)
        for layer, neuron in inhibited_neurons:
            if 0 <= layer < n_layers and 0 <= neuron < d_mlp:
                inhibited_mask[layer, neuron] = True

        # 损伤扩散（显著下降）
        damage_spread = []
        for layer in range(n_layers):
            for neuron in range(d_mlp):
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

        # 代偿激活（显著上升）
        compensation = []
        for layer in range(n_layers):
            for neuron in range(d_mlp):
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

        # 找出无变化的神经元（在监测层内，但既不是损伤也不是代偿）
        damage_set = {(d['layer'], d['neuron']) for d in damage_spread}
        comp_set = {(c['layer'], c['neuron']) for c in compensation}

        unchanged = []
        for layer in range(n_layers):
            for neuron in range(d_mlp):
                if inhibited_mask[layer, neuron]:
                    continue
                if (layer, neuron) not in damage_set and (layer, neuron) not in comp_set:
                    # 只记录那些差异在一定范围内的
                    diff = abs(intervention[layer, neuron] - baseline[layer, neuron])
                    if diff < threshold * std_baseline * 0.5:
                        unchanged.append({
                            'layer': layer,
                            'neuron': neuron,
                            'baseline': float(baseline[layer, neuron]),
                            'intervention': float(intervention[layer, neuron]),
                            'diff': float(diff)
                        })

        # 排序
        damage_spread.sort(key=lambda x: x['drop'], reverse=True)
        compensation.sort(key=lambda x: x['rise'], reverse=True)
        unchanged.sort(key=lambda x: x['diff'], reverse=True)

        return {
            'inhibition_results': inhibition_results,
            'damage_spread': damage_spread[:500],  # 限制数量
            'compensation': compensation[:500],
            'unchanged': unchanged[:500],
            'statistics': {
                'inhibited_count': len(inhibited_neurons),
                'damage_count': len(damage_spread),
                'compensation_count': len(compensation),
                'unchanged_count': len(unchanged),
                'threshold': threshold,
                'std_baseline': float(std_baseline),
                'mean_baseline': float(mean_baseline)
            }
        }


# ==================== CSV生成器 ====================
class CSVGenerator:
    """生成可下载的CSV文件"""

    @staticmethod
    def generate_summary_csv(all_results: List[Dict]) -> str:
        """生成汇总CSV"""
        rows = []
        for res in all_results:
            target = res['target_neuron']
            stats = res['statistics']
            csv_info = res.get('csv_info', {})

            inhib_success = all([r['success'] for r in res['inhibition_results']])

            rows.append({
                '序号': target['index'],
                '层': target['layer'],
                '神经元': target['neuron'],
                'CSV差异值': f"{csv_info.get('difference', 0):.6f}",
                '抑制成功': '是' if inhib_success else '否',
                '损伤扩散数': stats['damage_count'],
                '代偿激活数': stats['compensation_count'],
                '无变化数': stats['unchanged_count'],
                '总影响神经元': stats['damage_count'] + stats['compensation_count']
            })

        df = pd.DataFrame(rows)
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8-sig') as f:
            df.to_csv(f.name, index=False)
            return f.name

    @staticmethod
    def generate_damage_csv(all_results: List[Dict]) -> str:
        """生成损伤扩散CSV"""
        rows = []
        for res in all_results:
            target = res['target_neuron']
            for d in res['damage_spread']:
                rows.append({
                    '抑制层': target['layer'],
                    '抑制神经元': target['neuron'],
                    '损伤层': d['layer'],
                    '损伤神经元': d['neuron'],
                    '基线值': f"{d['baseline']:.6f}",
                    '干预值': f"{d['intervention']:.6f}",
                    '下降幅度': f"{d['drop']:.6f}",
                    '下降百分比': f"{d['drop_percent']:.2f}",
                    'Z分数': f"{d['z_score']:.2f}"
                })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['抑制层', '抑制神经元', '损伤层', '损伤神经元', '基线值', '干预值', '下降幅度', '下降百分比', 'Z分数'])
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8-sig') as f:
            df.to_csv(f.name, index=False)
            return f.name

    @staticmethod
    def generate_compensation_csv(all_results: List[Dict]) -> str:
        """生成代偿激活CSV"""
        rows = []
        for res in all_results:
            target = res['target_neuron']
            for c in res['compensation']:
                rows.append({
                    '抑制层': target['layer'],
                    '抑制神经元': target['neuron'],
                    '代偿层': c['layer'],
                    '代偿神经元': c['neuron'],
                    '基线值': f"{c['baseline']:.6f}",
                    '干预值': f"{c['intervention']:.6f}",
                    '上升幅度': f"{c['rise']:.6f}",
                    '上升百分比': f"{c['rise_percent']:.2f}",
                    'Z分数': f"{c['z_score']:.2f}"
                })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['抑制层', '抑制神经元', '代偿层', '代償神经元', '基线值', '干预值', '上升幅度', '上升百分比', 'Z分数'])
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8-sig') as f:
            df.to_csv(f.name, index=False)
            return f.name

    @staticmethod
    def generate_unchanged_csv(all_results: List[Dict]) -> str:
        """生成无变化神经元CSV"""
        rows = []
        for res in all_results:
            target = res['target_neuron']
            for u in res['unchanged']:
                rows.append({
                    '抑制层': target['layer'],
                    '抑制神经元': target['neuron'],
                    '层': u['layer'],
                    '神经元': u['neuron'],
                    '基线值': f"{u['baseline']:.6f}",
                    '干预值': f"{u['intervention']:.6f}",
                    '差异值': f"{u['diff']:.6f}"
                })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['抑制层', '抑制神经元', '层', '神经元', '基线值', '干预值', '差异值'])
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8-sig') as f:
            df.to_csv(f.name, index=False)
            return f.name

    @staticmethod
    def generate_all_in_one_csv(all_results: List[Dict]) -> str:
        """生成包含所有信息的单个CSV"""
        rows = []
        for res in all_results:
            target = res['target_neuron']
            stats = res['statistics']
            csv_info = res.get('csv_info', {})

            # 添加汇总行
            rows.append({
                '类型': '汇总',
                '抑制层': target['layer'],
                '抑制神经元': target['neuron'],
                '相关层': '',
                '相关神经元': '',
                '数值': '',
                '备注': f"损伤:{stats['damage_count']}, 代偿:{stats['compensation_count']}, 无变化:{stats['unchanged_count']}, CSV差异:{csv_info.get('difference', 0):.4f}"
            })

            # 添加损伤扩散
            for d in res['damage_spread'][:10]:  # 只取前10个
                rows.append({
                    '类型': '损伤',
                    '抑制层': target['layer'],
                    '抑制神经元': target['neuron'],
                    '相关层': d['layer'],
                    '相关神经元': d['neuron'],
                    '数值': f"{d['drop']:.6f}",
                    '备注': f"下降 {d['drop_percent']:.1f}%"
                })

            # 添加代偿激活
            for c in res['compensation'][:10]:
                rows.append({
                    '类型': '代偿',
                    '抑制层': target['layer'],
                    '抑制神经元': target['neuron'],
                    '相关层': c['layer'],
                    '相关神经元': c['neuron'],
                    '数值': f"{c['rise']:.6f}",
                    '备注': f"上升 {c['rise_percent']:.1f}%"
                })

        df = pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8-sig') as f:
            df.to_csv(f.name, index=False)
            return f.name


# ==================== Gradio界面 ====================
class AutoInterventionUI:
    """自动化神经元干预测试界面"""

    def __init__(self):
        print("🤖 初始化自动化神经元测试系统...")
        self.model = None
        self.analyzer = AnalysisEngine()
        self.csv_gen = CSVGenerator()
        self.last_results = None

    def load_model(self):
        """加载模型（延迟加载）"""
        if self.model is None:
            self.model = ModelManager()
        return self.model

    def run_auto_test(self,
                      csv_file,
                      test_sentence,
                      top_per_layer,
                      threshold,
                      progress=gr.Progress()):
        """运行自动化测试"""
        try:
            # 检查输入
            if csv_file is None:
                return "请上传CSV文件", None, None, None, None, None, None, None, None, None

            # 加载模型
            progress(0.1, desc="加载模型中...")
            model = self.load_model()

            # 读取CSV文件
            progress(0.2, desc="读取CSV文件...")
            df = pd.read_csv(csv_file.name)

            # 打印列名以便调试
            print(f"CSV列名: {df.columns.tolist()}")

            # 找出每层前N个差异最大的神经元
            target_neurons = []
            layers = sorted(df['layer'].unique())

            neuron_info = []
            for layer in layers:
                layer_data = df[df['layer'] == layer]
                layer_data = layer_data.sort_values('difference', ascending=False)
                for _, row in layer_data.head(top_per_layer).iterrows():
                    target_neurons.append((int(row['layer']), int(row['neuron'])))
                    # 只保存必要的字段，不依赖percent_change
                    info = {
                        'layer': int(row['layer']),
                        'neuron': int(row['neuron']),
                        'difference': float(row['difference'])
                    }
                    # 如果存在percent_change才添加
                    if 'percent_change' in row:
                        info['percent_change'] = float(row['percent_change'])
                    neuron_info.append(info)

            if not target_neurons:
                return "未找到目标神经元", None, None, None, None, None, None, None, None, None

            progress(0.3, desc=f"找到 {len(target_neurons)} 个目标神经元，收集基线激活值...")

            # 获取基线激活值（只做一次）
            baseline = model.get_baseline_activations(test_sentence)

            # 存储所有结果
            all_results = []

            # 逐个测试
            for idx, (layer, neuron) in enumerate(target_neurons):
                progress_val = 0.3 + 0.6 * (idx + 1) / len(target_neurons)
                progress(progress_val, desc=f"测试 {idx+1}/{len(target_neurons)}: 抑制 L{layer} N{neuron}")

                # 清理内存
                cleanup_memory()

                # 收集干预激活值
                intervention = model.get_intervention_activations(test_sentence, layer, neuron)

                # 分析结果
                results = self.analyzer.analyze(
                    baseline=baseline,
                    intervention=intervention,
                    inhibited_neurons=[(layer, neuron)],
                    threshold=threshold
                )

                # 添加神经元信息
                results['target_neuron'] = {
                    'layer': layer,
                    'neuron': neuron,
                    'index': idx + 1
                }

                # 从CSV中找对应的差异信息
                for info in neuron_info:
                    if info['layer'] == layer and info['neuron'] == neuron:
                        results['csv_info'] = info
                        break

                all_results.append(results)

                # 再次清理
                cleanup_memory()
                time.sleep(0.5)  # 短暂延迟

            self.last_results = all_results

            progress(1.0, desc="生成CSV文件...")

            # 生成所有CSV文件
            summary_csv = self.csv_gen.generate_summary_csv(all_results)
            damage_csv = self.csv_gen.generate_damage_csv(all_results)
            comp_csv = self.csv_gen.generate_compensation_csv(all_results)
            unchanged_csv = self.csv_gen.generate_unchanged_csv(all_results)
            all_in_one_csv = self.csv_gen.generate_all_in_one_csv(all_results)

            # 生成预览表格
            summary_df = pd.read_csv(summary_csv)
            damage_df = pd.read_csv(damage_csv) if os.path.getsize(damage_csv) > 0 else pd.DataFrame()
            comp_df = pd.read_csv(comp_csv) if os.path.getsize(comp_csv) > 0 else pd.DataFrame()
            unchanged_df = pd.read_csv(unchanged_csv) if os.path.getsize(unchanged_csv) > 0 else pd.DataFrame()

            return (
                f"✅ 测试完成！共测试 {len(target_neurons)} 个神经元",
                summary_df,
                damage_df,
                comp_df,
                unchanged_df,
                summary_csv,
                damage_csv,
                comp_csv,
                unchanged_csv,
                all_in_one_csv
            )

        except Exception as e:
            traceback.print_exc()
            return f"❌ 错误: {str(e)}", None, None, None, None, None, None, None, None, None

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="🤖 自动化神经元干预测试", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🤖 自动化神经元干预测试系统")
            gr.Markdown("### 上传CSV文件 → 自动测试所有显著差异神经元 → 下载CSV结果")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **模型信息**
                    - 模型: Llama2 7B (4bit)
                    - 层数: 0-31
                    - 神经元/层: 0-11007
                    """)

            gr.Markdown("---")

            with gr.Row():
                with gr.Column(scale=1):
                    # 输入区域
                    csv_input = gr.File(
                        label="📁 上传差异神经元CSV文件",
                        file_types=[".csv"]
                    )

                    sentence_input = gr.Textbox(
                        label="📝 测试句子",
                        value="Explain how neural networks work.",
                        lines=2,
                        placeholder="输入要测试的句子..."
                    )

                    with gr.Row():
                        top_n = gr.Slider(
                            label="每层取前N个神经元",
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1
                        )

                        threshold = gr.Slider(
                            label="显著阈值 (标准差倍数)",
                            minimum=1.0,
                            maximum=5.0,
                            value=2.0,
                            step=0.5
                        )

                    run_btn = gr.Button("🚀 开始自动化测试", variant="primary", size="lg")

                    status_text = gr.Textbox(label="状态", lines=2, interactive=False)

                with gr.Column(scale=2):
                    # 结果显示区域
                    with gr.Tabs():
                        with gr.TabItem("📊 汇总预览"):
                            summary_table = gr.Dataframe(
                                label="每个测试神经元的统计结果",
                                interactive=False,
                                wrap=True
                            )

                        with gr.TabItem("📉 损伤扩散预览"):
                            damage_table = gr.Dataframe(
                                label="所有被拖累的神经元",
                                interactive=False,
                                wrap=True
                            )

                        with gr.TabItem("📈 代偿激活预览"):
                            comp_table = gr.Dataframe(
                                label="所有代偿的神经元",
                                interactive=False,
                                wrap=True
                            )

                        with gr.TabItem("⚪ 无变化预览"):
                            unchanged_table = gr.Dataframe(
                                label="无明显变化的神经元",
                                interactive=False,
                                wrap=True
                            )

            gr.Markdown("---")
            gr.Markdown("## 📥 下载CSV文件")

            with gr.Row():
                with gr.Column():
                    summary_download = gr.File(
                        label="📊 下载汇总结果",
                        file_types=[".csv"],
                        interactive=False
                    )
                    damage_download = gr.File(
                        label="📉 下载损伤扩散",
                        file_types=[".csv"],
                        interactive=False
                    )

                with gr.Column():
                    comp_download = gr.File(
                        label="📈 下载代偿激活",
                        file_types=[".csv"],
                        interactive=False
                    )
                    unchanged_download = gr.File(
                        label="⚪ 下载无变化",
                        file_types=[".csv"],
                        interactive=False
                    )

            with gr.Row():
                all_in_one_download = gr.File(
                    label="📁 下载合并报告",
                    file_types=[".csv"],
                    interactive=False
                )

            # 绑定事件
            run_btn.click(
                fn=self.run_auto_test,
                inputs=[csv_input, sentence_input, top_n, threshold],
                outputs=[
                    status_text,
                    summary_table, damage_table, comp_table, unchanged_table,
                    summary_download, damage_download, comp_download,
                    unchanged_download, all_in_one_download
                ]
            )

        return app


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("🤖 自动化神经元干预测试系统 - 网页界面")
    print("=" * 70)
    print("功能说明:")
    print("  1. 上传CSV文件（包含显著差异神经元）")
    print("  2. 输入测试句子")
    print("  3. 设置每层取前N个神经元")
    print("  4. 自动测试每个神经元")
    print("  5. 下载CSV结果文件")
    print("=" * 70)

    ui = AutoInterventionUI()
    app = ui.create_interface()

    print("\n🌐 启动Gradio界面...")
    print("🔗 本地地址: http://127.0.0.1:7866")
    print("📁 测试完成后，点击下方按钮下载CSV文件")
    print("=" * 70)

    app.launch(
        server_name="127.0.0.1",
        server_port=7866,
        share=False,
        quiet=False
    )


if __name__ == "__main__":
    main()