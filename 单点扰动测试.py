import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gradio as gr
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ==================== 神经元干预器 ====================
class NeuronIntervention:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.intervention_history = []
        self.current_interventions = {}  # {(layer, neuron): intervention_type}
        self._load_model()

    def _load_model(self):
        """加载模型"""
        print(f"Loading model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)

        self.model.eval()
        print("Model loaded successfully")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if not self.model:
            return {}

        info = {
            "model_name": os.path.basename(self.model_path),
            "num_layers": len(self.model.gpt_neox.layers),
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype)
        }

        # 获取FFN层大小信息（假设第一层有代表性）
        if len(self.model.gpt_neox.layers) > 0:
            first_layer = self.model.gpt_neox.layers[0]
            ffn = first_layer.mlp
            # 获取激活函数后的维度
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, self.model.config.hidden_size)
                output = ffn.act(dummy_input)
                info["ffn_dim"] = output.shape[-1]
                info["hidden_size"] = self.model.config.hidden_size

        return info

    def _setup_intervention_hooks(self):
        """设置干预钩子"""
        self._remove_hooks()  # 先清理现有钩子

        def make_intervention_hook(layer_idx, neuron_idx, intervention_type, strength=1.0):
            def hook(module, input, output):
                # output形状: [batch, seq_len, ffn_dim]
                if output is not None:
                    # 克隆输出以避免原地修改问题
                    modified_output = output.clone()

                    if intervention_type == "inhibit":
                        # 抑制：将指定神经元设为0
                        modified_output[:, :, neuron_idx] = 0
                    elif intervention_type == "activate":
                        # 激活：增强指定神经元
                        modified_output[:, :, neuron_idx] = modified_output[:, :, neuron_idx] * (1.0 + strength)
                    elif intervention_type == "set_value":
                        # 设置为特定值
                        modified_output[:, :, neuron_idx] = strength

                    return modified_output

            return hook

        # 为每个干预设置钩子
        for (layer_idx, neuron_idx), intervention_info in self.current_interventions.items():
            intervention_type = intervention_info["type"]
            strength = intervention_info.get("strength", 1.0)

            if 0 <= layer_idx < len(self.model.gpt_neox.layers):
                layer = self.model.gpt_neox.layers[layer_idx]
                ffn_module = layer.mlp.act  # 在激活函数后干预

                hook = ffn_module.register_forward_hook(
                    make_intervention_hook(layer_idx, neuron_idx, intervention_type, strength)
                )
                self.hooks.append(hook)
                print(f"Hook set: Layer {layer_idx}, Neuron {neuron_idx}, Type: {intervention_type}")

    def _remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def add_intervention(self, layer_idx: int, neuron_idx: int,
                         intervention_type: str, strength: float = 1.0):
        """添加神经元干预"""
        key = (layer_idx, neuron_idx)

        # 验证参数
        model_info = self.get_model_info()
        num_layers = model_info.get("num_layers", 0)
        ffn_dim = model_info.get("ffn_dim", 0)

        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer index must be between 0 and {num_layers - 1}")

        if neuron_idx < 0 or neuron_idx >= ffn_dim:
            raise ValueError(f"Neuron index must be between 0 and {ffn_dim - 1}")

        if intervention_type not in ["inhibit", "activate", "set_value"]:
            raise ValueError("Intervention type must be 'inhibit', 'activate', or 'set_value'")

        # 添加干预
        self.current_interventions[key] = {
            "type": intervention_type,
            "strength": strength,
            "added_time": time.time()
        }

        # 重新设置钩子
        self._setup_intervention_hooks()

        return f"Added intervention: Layer {layer_idx}, Neuron {neuron_idx}, Type: {intervention_type}"

    def remove_intervention(self, layer_idx: int, neuron_idx: int):
        """移除特定干预"""
        key = (layer_idx, neuron_idx)
        if key in self.current_interventions:
            del self.current_interventions[key]
            self._setup_intervention_hooks()  # 重新设置钩子
            return f"Removed intervention: Layer {layer_idx}, Neuron {neuron_idx}"
        return f"No intervention found for Layer {layer_idx}, Neuron {neuron_idx}"

    def clear_all_interventions(self):
        """清除所有干预"""
        self.current_interventions.clear()
        self._remove_hooks()
        return "Cleared all interventions"

    def get_current_interventions(self) -> List[Dict]:
        """获取当前干预列表"""
        interventions = []
        for (layer, neuron), info in self.current_interventions.items():
            interventions.append({
                "layer": layer,
                "neuron": neuron,
                "type": info["type"],
                "strength": info.get("strength", 1.0)
            })
        return interventions

    def generate_with_intervention(self, prompt: str, max_length: int = 100) -> Dict:
        """在干预下生成文本"""
        try:
            # 记录开始时间
            start_time = time.time()

            # 准备输入
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 提取新生成的部分
            prompt_tokens = len(self.tokenizer.encode(prompt))
            response_tokens = self.tokenizer.encode(generated_text)[prompt_tokens:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            # 创建记录
            record = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": response_text,
                "full_output": generated_text,
                "interventions": self.get_current_interventions(),
                "generation_time": time.time() - start_time,
                "response_length": len(response_text)
            }

            # 添加到历史记录
            self.intervention_history.append(record)

            return {
                "success": True,
                "response": response_text,
                "full_output": generated_text,
                "record": record
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"Error: {str(e)}"
            }

    def get_history(self) -> List[Dict]:
        """获取历史记录"""
        return self.intervention_history.copy()

    def clear_history(self):
        """清除历史记录"""
        self.intervention_history.clear()
        return "History cleared"

    def save_history(self, filename: str = None):
        """保存历史记录到JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"intervention_history_{timestamp}.json"

        output_path = Path("intervention_results") / filename
        output_path.parent.mkdir(exist_ok=True)

        # 准备可序列化的数据
        serializable_history = []
        for record in self.intervention_history:
            serializable_record = record.copy()
            # 确保所有数据可JSON序列化
            serializable_record["timestamp"] = str(serializable_record["timestamp"])
            serializable_history.append(serializable_record)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def export_history_csv(self, filename: str = None):
        """导出历史记录到CSV"""
        if not self.intervention_history:
            return "No history to export"

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"intervention_history_{timestamp}.csv"

        output_path = Path("intervention_results") / filename
        output_path.parent.mkdir(exist_ok=True)

        # 准备DataFrame
        records = []
        for i, record in enumerate(self.intervention_history):
            row = {
                "id": i + 1,
                "timestamp": record["timestamp"],
                "prompt": record["prompt"],
                "response": record["response"],
                "generation_time": record["generation_time"],
                "response_length": record["response_length"]
            }

            # 添加干预信息
            interventions = record["interventions"]
            for j, interv in enumerate(interventions):
                row[f"intervention_{j + 1}_layer"] = interv["layer"]
                row[f"intervention_{j + 1}_neuron"] = interv["neuron"]
                row[f"intervention_{j + 1}_type"] = interv["type"]
                row[f"intervention_{j + 1}_strength"] = interv.get("strength", 1.0)

            records.append(row)

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False, encoding='utf-8')

        return str(output_path)


# ==================== Gradio界面 ====================
class InterventionInterface:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.intervention = NeuronIntervention(model_path)
        self.model_info = self.intervention.get_model_info()

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="Neuron Intervention Analyzer", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🧠 Neuron Intervention Analysis Tool")
            gr.Markdown(f"Model: {self.model_info.get('model_name', 'Unknown')} | "
                        f"Layers: {self.model_info.get('num_layers', 'N/A')} | "
                        f"FFN Dim: {self.model_info.get('ffn_dim', 'N/A')}")

            # === 左侧：干预控制 ===
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 🎯 Neuron Intervention Controls")

                    with gr.Group():
                        layer_input = gr.Number(
                            label="Layer Index",
                            value=0,
                            minimum=0,
                            maximum=self.model_info.get('num_layers', 12) - 1,
                            step=1,
                            precision=0
                        )

                        neuron_input = gr.Number(
                            label="Neuron Index",
                            value=0,
                            minimum=0,
                            maximum=self.model_info.get('ffn_dim', 3072) - 1,
                            step=1,
                            precision=0
                        )

                    intervention_type = gr.Radio(
                        label="Intervention Type",
                        choices=["inhibit", "activate", "set_value"],
                        value="inhibit"
                    )

                    strength_input = gr.Slider(
                        label="Strength (for activate/set_value)",
                        minimum=0.1,
                        maximum=10.0,
                        value=1.0,
                        step=0.1,
                        visible=False
                    )

                    # 根据干预类型显示/隐藏强度滑块
                    def toggle_strength(choice):
                        return gr.Slider(visible=choice in ["activate", "set_value"])

                    intervention_type.change(
                        toggle_strength,
                        inputs=[intervention_type],
                        outputs=[strength_input]
                    )

                    with gr.Row():
                        add_btn = gr.Button("➕ Add Intervention", variant="primary")
                        remove_btn = gr.Button("➖ Remove Intervention")

                    clear_all_btn = gr.Button("🗑️ Clear All Interventions", variant="secondary")

                    # 当前干预列表
                    current_interventions = gr.DataFrame(
                        label="Current Interventions",
                        headers=["Layer", "Neuron", "Type", "Strength"],
                        interactive=False
                    )

                    refresh_btn = gr.Button("🔄 Refresh List")

                # === 中间：文本生成 ===
                with gr.Column(scale=2):
                    gr.Markdown("## 💬 Text Generation")

                    prompt_input = gr.Textbox(
                        label="Input Prompt",
                        value="Explain the concept of artificial intelligence:",
                        lines=4
                    )

                    max_length = gr.Slider(
                        label="Max Response Length",
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10
                    )

                    generate_btn = gr.Button("🚀 Generate Response", variant="primary")

                    response_output = gr.Textbox(
                        label="Model Response",
                        lines=6,
                        interactive=False
                    )

                    full_output = gr.Textbox(
                        label="Full Output (for debugging)",
                        lines=3,
                        interactive=False,
                        visible=False
                    )

                    show_debug = gr.Checkbox(label="Show full output", value=False)

                    # 切换调试输出显示
                    def toggle_debug(show):
                        return gr.Textbox(visible=show)

                    show_debug.change(
                        toggle_debug,
                        inputs=[show_debug],
                        outputs=[full_output]
                    )

                # === 右侧：历史记录 ===
                with gr.Column(scale=1):
                    gr.Markdown("## 📊 History & Analysis")

                    history_df = gr.DataFrame(
                        label="Generation History",
                        headers=["Time", "Prompt", "Response", "Interventions"],
                        interactive=False,
                        wrap=True
                    )

                    with gr.Row():
                        clear_history_btn = gr.Button("🗑️ Clear History")
                        save_json_btn = gr.Button("💾 Save as JSON")
                        save_csv_btn = gr.Button("📊 Save as CSV")

                    saved_files = gr.File(
                        label="Saved Files",
                        file_count="multiple"
                    )

                    # 文件浏览器
                    def list_saved_files():
                        results_dir = Path("intervention_results")
                        if not results_dir.exists():
                            return []
                        return [str(f) for f in results_dir.glob("*.json")] + \
                            [str(f) for f in results_dir.glob("*.csv")]

                    refresh_files_btn = gr.Button("🔄 Refresh Files")

                    refresh_files_btn.click(
                        fn=list_saved_files,
                        outputs=[saved_files]
                    )

            # === 状态信息 ===
            status_output = gr.Textbox(
                label="Status",
                value="Ready",
                interactive=False
            )

            # === 事件处理 ===
            # 添加干预
            add_btn.click(
                fn=self.add_intervention,
                inputs=[layer_input, neuron_input, intervention_type, strength_input],
                outputs=[status_output]
            ).then(
                fn=self.get_current_interventions,
                outputs=[current_interventions]
            )

            # 移除干预
            remove_btn.click(
                fn=self.remove_intervention,
                inputs=[layer_input, neuron_input],
                outputs=[status_output]
            ).then(
                fn=self.get_current_interventions,
                outputs=[current_interventions]
            )

            # 清除所有干预
            clear_all_btn.click(
                fn=self.clear_interventions,
                outputs=[status_output]
            ).then(
                fn=self.get_current_interventions,
                outputs=[current_interventions]
            )

            # 刷新干预列表
            refresh_btn.click(
                fn=self.get_current_interventions,
                outputs=[current_interventions]
            )

            # 生成文本
            generate_btn.click(
                fn=self.generate_response,
                inputs=[prompt_input, max_length],
                outputs=[response_output, full_output, status_output]
            ).then(
                fn=self.get_history_df,
                outputs=[history_df]
            )

            # 历史记录管理
            clear_history_btn.click(
                fn=self.clear_history,
                outputs=[status_output]
            ).then(
                fn=self.get_history_df,
                outputs=[history_df]
            )

            save_json_btn.click(
                fn=self.save_history_json,
                outputs=[status_output]
            ).then(
                fn=list_saved_files,
                outputs=[saved_files]
            )

            save_csv_btn.click(
                fn=self.save_history_csv,
                outputs=[status_output]
            ).then(
                fn=list_saved_files,
                outputs=[saved_files]
            )

            # 初始加载
            app.load(
                fn=self.get_current_interventions,
                outputs=[current_interventions]
            ).then(
                fn=self.get_history_df,
                outputs=[history_df]
            ).then(
                fn=list_saved_files,
                outputs=[saved_files]
            )

        return app

    # === 包装器方法 ===
    def add_intervention(self, layer_idx: int, neuron_idx: int,
                         intervention_type: str, strength: float):
        try:
            result = self.intervention.add_intervention(
                int(layer_idx), int(neuron_idx), intervention_type, float(strength)
            )
            return f"✅ {result}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def remove_intervention(self, layer_idx: int, neuron_idx: int):
        try:
            result = self.intervention.remove_intervention(int(layer_idx), int(neuron_idx))
            return f"✅ {result}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def clear_interventions(self):
        try:
            result = self.intervention.clear_all_interventions()
            return f"✅ {result}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_current_interventions(self):
        interventions = self.intervention.get_current_interventions()
        if not interventions:
            return pd.DataFrame(columns=["Layer", "Neuron", "Type", "Strength"])

        df_data = []
        for interv in interventions:
            df_data.append([
                interv["layer"],
                interv["neuron"],
                interv["type"],
                interv.get("strength", 1.0)
            ])

        return pd.DataFrame(df_data, columns=["Layer", "Neuron", "Type", "Strength"])

    def generate_response(self, prompt: str, max_length: int):
        result = self.intervention.generate_with_intervention(prompt, int(max_length))

        if result["success"]:
            status = f"✅ Generated {len(result['response'])} characters"
            return result["response"], result["full_output"], status
        else:
            return f"Error: {result['error']}", "", f"❌ {result['error']}"

    def get_history_df(self):
        history = self.intervention.get_history()
        if not history:
            return pd.DataFrame(columns=["Time", "Prompt", "Response", "Interventions"])

        df_data = []
        for record in history[-20:]:  # 显示最近20条
            # 格式化时间
            time_str = datetime.fromisoformat(record["timestamp"]).strftime("%H:%M:%S")

            # 格式化干预信息
            interv_str = ", ".join([
                f"L{i['layer']}N{i['neuron']}({i['type'][0]})"
                for i in record["interventions"]
            ]) if record["interventions"] else "None"

            # 截断长文本
            prompt_preview = record["prompt"][:50] + "..." if len(record["prompt"]) > 50 else record["prompt"]
            response_preview = record["response"][:100] + "..." if len(record["response"]) > 100 else record["response"]

            df_data.append([time_str, prompt_preview, response_preview, interv_str])

        return pd.DataFrame(df_data, columns=["Time", "Prompt", "Response", "Interventions"])

    def clear_history(self):
        result = self.intervention.clear_history()
        return f"✅ {result}"

    def save_history_json(self):
        try:
            filepath = self.intervention.save_history()
            return f"✅ History saved to: {os.path.basename(filepath)}"
        except Exception as e:
            return f"❌ Save failed: {str(e)}"

    def save_history_csv(self):
        try:
            filepath = self.intervention.export_history_csv()
            return f"✅ History exported to: {os.path.basename(filepath)}"
        except Exception as e:
            return f"❌ Export failed: {str(e)}"


# ==================== 主程序 ====================
def main():
    MODEL_PATH = r"E:\AI_Models\Pythia-160M\models"

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Warning: Model path does not exist: {MODEL_PATH}")
        print("Please ensure the model is downloaded to the correct path")
        MODEL_PATH = input("Enter correct model path (or press Enter to use default): ") or MODEL_PATH

    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model path does not exist, exiting...")
        return

    print("🚀 Starting Neuron Intervention Analyzer...")
    print(f"📁 Model path: {MODEL_PATH}")

    app = InterventionInterface(MODEL_PATH)
    interface = app.create_interface()

    print("🌐 Gradio interface starting...")
    print("👉 Local access: http://127.0.0.1:7861")
    print("👉 Press Ctrl+C to stop")

    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=7861,  # 使用不同端口避免冲突
            share=False,
            quiet=False
        )
    except Exception as e:
        print(f"❌ Launch failed: {e}")


if __name__ == "__main__":
    main()