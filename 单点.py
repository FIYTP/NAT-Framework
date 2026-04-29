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


# ==================== Neuron Intervention System ====================
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
        """Load model"""
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
        """Get model information"""
        if not self.model:
            return {}

        info = {
            "model_name": os.path.basename(self.model_path),
            "num_layers": len(self.model.gpt_neox.layers),
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "hidden_size": self.model.config.hidden_size
        }

        # Get FFN dimension information for Pythia-410M
        # Pythia-410M has 24 layers, 1024 hidden size, 4096 FFN dimension
        info["ffn_dim"] = 4096  # Fixed for Pythia-410M

        return info

    def _setup_intervention_hooks(self):
        """Setup intervention hooks"""
        self._remove_hooks()  # Clear existing hooks first

        def make_intervention_hook(layer_idx, neuron_idx, intervention_type, strength=1.0):
            def hook(module, input, output):
                # output shape: [batch, seq_len, ffn_dim]
                if output is not None:
                    # Clone output to avoid in-place modification issues
                    modified_output = output.clone()

                    if intervention_type == "inhibit":
                        # Inhibition: set specified neuron to 0
                        modified_output[:, :, neuron_idx] = 0
                    elif intervention_type == "activate":
                        # Activation: enhance specified neuron
                        modified_output[:, :, neuron_idx] = modified_output[:, :, neuron_idx] * (1.0 + strength)
                    elif intervention_type == "set_value":
                        # Set to specific value
                        modified_output[:, :, neuron_idx] = strength

                    return modified_output

            return hook

        # Set hooks for each intervention
        for (layer_idx, neuron_idx), intervention_info in self.current_interventions.items():
            intervention_type = intervention_info["type"]
            strength = intervention_info.get("strength", 1.0)

            if 0 <= layer_idx < len(self.model.gpt_neox.layers):
                layer = self.model.gpt_neox.layers[layer_idx]
                ffn_module = layer.mlp.act  # Intervene after activation function

                hook = ffn_module.register_forward_hook(
                    make_intervention_hook(layer_idx, neuron_idx, intervention_type, strength)
                )
                self.hooks.append(hook)
                print(f"Hook set: Layer {layer_idx}, Neuron {neuron_idx}, Type: {intervention_type}")

    def _remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def add_intervention(self, layer_idx: int, neuron_idx: int,
                         intervention_type: str, strength: float = 1.0):
        """Add neuron intervention"""
        key = (layer_idx, neuron_idx)

        # Validate parameters
        model_info = self.get_model_info()
        num_layers = model_info.get("num_layers", 24)  # 410M has 24 layers
        ffn_dim = model_info.get("ffn_dim", 4096)  # 410M has 4096 neurons per layer

        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer index must be between 0 and {num_layers - 1}")

        if neuron_idx < 0 or neuron_idx >= ffn_dim:
            raise ValueError(f"Neuron index must be between 0 and {ffn_dim - 1}")

        if intervention_type not in ["inhibit", "activate", "set_value"]:
            raise ValueError("Intervention type must be 'inhibit', 'activate', or 'set_value'")

        # Add intervention
        self.current_interventions[key] = {
            "type": intervention_type,
            "strength": strength,
            "added_time": time.time()
        }

        # Re-setup hooks
        self._setup_intervention_hooks()

        return f"Added intervention: Layer {layer_idx}, Neuron {neuron_idx}, Type: {intervention_type}"

    def remove_intervention(self, layer_idx: int, neuron_idx: int):
        """Remove specific intervention"""
        key = (layer_idx, neuron_idx)
        if key in self.current_interventions:
            del self.current_interventions[key]
            self._setup_intervention_hooks()  # Re-setup hooks
            return f"Removed intervention: Layer {layer_idx}, Neuron {neuron_idx}"
        return f"No intervention found for Layer {layer_idx}, Neuron {neuron_idx}"

    def clear_all_interventions(self):
        """Clear all interventions"""
        self.current_interventions.clear()
        self._remove_hooks()
        return "Cleared all interventions"

    def get_current_interventions(self) -> List[Dict]:
        """Get current intervention list"""
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
        """Generate text with intervention"""
        try:
            # Record start time
            start_time = time.time()

            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate text
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

            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract newly generated part
            prompt_tokens = len(self.tokenizer.encode(prompt))
            response_tokens = self.tokenizer.encode(generated_text)[prompt_tokens:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            # Create record
            record = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": response_text,
                "full_output": generated_text,
                "interventions": self.get_current_interventions(),
                "generation_time": time.time() - start_time,
                "response_length": len(response_text)
            }

            # Add to history
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
        """Get history records"""
        return self.intervention_history.copy()

    def clear_history(self):
        """Clear history"""
        self.intervention_history.clear()
        return "History cleared"

    def save_history(self, filename: str = None):
        """Save history to JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"intervention_history_{timestamp}.json"

        output_path = Path("intervention_results") / filename
        output_path.parent.mkdir(exist_ok=True)

        # Prepare serializable data
        serializable_history = []
        for record in self.intervention_history:
            serializable_record = record.copy()
            # Ensure all data is JSON serializable
            serializable_record["timestamp"] = str(serializable_record["timestamp"])
            serializable_history.append(serializable_record)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def export_history_csv(self, filename: str = None):
        """Export history to CSV"""
        if not self.intervention_history:
            return "No history to export"

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"intervention_history_{timestamp}.csv"

        output_path = Path("intervention_results") / filename
        output_path.parent.mkdir(exist_ok=True)

        # Prepare DataFrame
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

            # Add intervention information
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


# ==================== Gradio Interface ====================
class InterventionInterface:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.intervention = NeuronIntervention(model_path)
        self.model_info = self.intervention.get_model_info()

    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(title="Pythia-410M Neuron Intervention Analyzer", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🧠 Pythia-410M Neuron Intervention Analysis Tool")
            gr.Markdown("### Single Neuron Perturbation Testing for Model Behavior Analysis")

            # Model info display
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"**Model:** {self.model_info.get('model_name', 'Pythia-410M')}")
                    gr.Markdown(f"**Layers:** {self.model_info.get('num_layers', 24)} (0-23)")
                    gr.Markdown(f"**FFN Dimension:** {self.model_info.get('ffn_dim', 4096)} neurons per layer")
                    gr.Markdown(f"**Hidden Size:** {self.model_info.get('hidden_size', 1024)}")
                    gr.Markdown(f"**Device:** {self.model_info.get('device', 'CPU/GPU')}")

            # === Left Column: Intervention Controls ===
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 🎯 Neuron Intervention Controls")

                    with gr.Group():
                        layer_input = gr.Number(
                            label="Layer Index",
                            value=10,
                            minimum=0,
                            maximum=self.model_info.get('num_layers', 24) - 1,
                            step=1,
                            precision=0,
                            info="Pythia-410M has 24 layers (0-23)"
                        )

                        neuron_input = gr.Number(
                            label="Neuron Index",
                            value=2048,
                            minimum=0,
                            maximum=self.model_info.get('ffn_dim', 4096) - 1,
                            step=1,
                            precision=0,
                            info="4096 neurons per layer (0-4095)"
                        )

                    intervention_type = gr.Radio(
                        label="Intervention Type",
                        choices=[
                            ("Inhibit (set to 0)", "inhibit"),
                            ("Activate (enhance)", "activate"),
                            ("Set Value", "set_value")
                        ],
                        value="inhibit"
                    )

                    strength_input = gr.Slider(
                        label="Strength (for activate/set_value)",
                        minimum=0.1,
                        maximum=10.0,
                        value=2.0,
                        step=0.1,
                        visible=False
                    )

                    # Show/hide strength slider based on intervention type
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

                    # Current interventions list
                    current_interventions = gr.DataFrame(
                        label="Current Active Interventions",
                        headers=["Layer", "Neuron", "Type", "Strength"],
                        interactive=False
                    )

                    refresh_btn = gr.Button("🔄 Refresh List")

                # === Middle Column: Text Generation ===
                with gr.Column(scale=2):
                    gr.Markdown("## 💬 Text Generation with Interventions")

                    prompt_input = gr.Textbox(
                        label="Input Prompt",
                        value="Explain how artificial intelligence works in simple terms:",
                        lines=5,
                        placeholder="Enter your prompt here..."
                    )

                    with gr.Row():
                        max_length = gr.Slider(
                            label="Max Response Length (tokens)",
                            minimum=10,
                            maximum=500,
                            value=150,
                            step=10
                        )

                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1
                        )

                    generate_btn = gr.Button("🚀 Generate with Interventions", variant="primary", size="lg")

                    response_output = gr.Textbox(
                        label="Model Response (with interventions)",
                        lines=8,
                        interactive=False
                    )

                    full_output = gr.Textbox(
                        label="Full Output (for debugging)",
                        lines=4,
                        interactive=False,
                        visible=False
                    )

                    show_debug = gr.Checkbox(label="Show full output", value=False)

                    # Toggle debug output visibility
                    def toggle_debug(show):
                        return gr.Textbox(visible=show)

                    show_debug.change(
                        toggle_debug,
                        inputs=[show_debug],
                        outputs=[full_output]
                    )

                # === Right Column: History & Analysis ===
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

                    # File browser
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

            # === Status Information ===
            status_output = gr.Textbox(
                label="Status",
                value="Ready to start neuron interventions on Pythia-410M",
                interactive=False
            )

            # === Event Handlers ===
            # Add intervention
            add_btn.click(
                fn=self.add_intervention,
                inputs=[layer_input, neuron_input, intervention_type, strength_input],
                outputs=[status_output]
            ).then(
                fn=self.get_current_interventions,
                outputs=[current_interventions]
            )

            # Remove intervention
            remove_btn.click(
                fn=self.remove_intervention,
                inputs=[layer_input, neuron_input],
                outputs=[status_output]
            ).then(
                fn=self.get_current_interventions,
                outputs=[current_interventions]
            )

            # Clear all interventions
            clear_all_btn.click(
                fn=self.clear_interventions,
                outputs=[status_output]
            ).then(
                fn=self.get_current_interventions,
                outputs=[current_interventions]
            )

            # Refresh interventions list
            refresh_btn.click(
                fn=self.get_current_interventions,
                outputs=[current_interventions]
            )

            # Generate text
            generate_btn.click(
                fn=self.generate_response,
                inputs=[prompt_input, max_length],
                outputs=[response_output, full_output, status_output]
            ).then(
                fn=self.get_history_df,
                outputs=[history_df]
            )

            # History management
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

            # Initial loading
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

    # === Wrapper Methods ===
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
            status = f"✅ Generated {len(result['response'])} characters with {len(self.intervention.current_interventions)} interventions"
            return result["response"], result["full_output"], status
        else:
            return f"Error: {result['error']}", "", f"❌ {result['error']}"

    def get_history_df(self):
        history = self.intervention.get_history()
        if not history:
            return pd.DataFrame(columns=["Time", "Prompt", "Response", "Interventions"])

        df_data = []
        for record in history[-20:]:  # Show last 20 records
            # Format time
            time_str = datetime.fromisoformat(record["timestamp"]).strftime("%H:%M:%S")

            # Format intervention information
            interv_str = ", ".join([
                f"L{i['layer']}N{i['neuron']}({i['type'][0]})"
                for i in record["interventions"]
            ]) if record["interventions"] else "None"

            # Truncate long text
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


# ==================== Main Program ====================
def main():
    # Updated for Pythia-410M model
    MODEL_PATH = r"E:\AI_Models\Pythia\Pythia-410M\models"

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Warning: Model path does not exist: {MODEL_PATH}")
        print("Please ensure the model is downloaded to the correct path")
        MODEL_PATH = input("Enter correct model path (or press Enter to use default): ") or MODEL_PATH

    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model path does not exist, exiting...")
        return

    print("=" * 60)
    print("🚀 Starting Pythia-410M Neuron Intervention Analyzer")
    print("=" * 60)
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"🔬 Target: Pythia-410M (24 layers, 4096 neurons/layer)")
    print(f"🎯 Purpose: Single neuron perturbation testing")
    print("=" * 60)

    app = InterventionInterface(MODEL_PATH)
    interface = app.create_interface()

    print("🌐 Gradio interface starting...")
    print("👉 Local access: http://127.0.0.1:7862")
    print("👉 Press Ctrl+C to stop")
    print("=" * 60)

    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=7862,  # Different port to avoid conflicts
            share=False,
            quiet=False
        )
    except Exception as e:
        print(f"❌ Launch failed: {e}")


if __name__ == "__main__":
    main()