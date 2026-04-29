"""
Neuron Intervention Experiment - Correct Four Module Structure (Pythia-410M)
FIXED: Changed dropdown to file explorer for fingerprint selection
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gradio as gr
import json
from pathlib import Path
import warnings
from datetime import datetime
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Any
import gc
import traceback

warnings.filterwarnings('ignore')


# ==================== Model Manager - Fixed Intervention (Pythia-410M) ====================
class ModelManager:
    """Manage local Pythia-410M model with working intervention"""

    def __init__(self, model_path: str = r"E:\AI_Models\Pythia\Pythia-410M\models"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.n_layers = 24
        self.d_mlp = 4096
        self._load_model()

    def _load_model(self):
        """Load local model"""
        print("=" * 60)
        print("🚀 Loading Pythia-410M from LOCAL path")
        print("=" * 60)

        try:
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

            if hasattr(self.model.config, 'num_hidden_layers'):
                self.n_layers = self.model.config.num_hidden_layers
            if hasattr(self.model.config, 'intermediate_size'):
                self.d_mlp = self.model.config.intermediate_size

            print(f"✅ Model loaded: {self.n_layers} layers, {self.d_mlp} neurons/layer")
            print(f"📊 Total neurons: {self.n_layers * self.d_mlp:,}")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def get_fingerprint(self,
                        sentences: List[str],
                        monitor_layers: List[int],
                        inhibit_neurons: List[Tuple[int, int]] = None,
                        repeats: int = 1) -> np.ndarray:
        """
        Generate fingerprint with WORKING intervention
        """
        if not sentences or not monitor_layers:
            return np.array([])

        layer_activations = {layer: [] for layer in monitor_layers}

        def create_inhibition_hook(layer_idx, neuron_idx):
            def hook(module, input, output):
                if output is not None:
                    modified_output = output.clone()
                    modified_output[..., neuron_idx] = 0.0
                    if torch.is_tensor(modified_output):
                        first_val = modified_output[0, 0, neuron_idx].item()
                        print(f"   🎯 Hook L{layer_idx} N{neuron_idx}: Set to 0 (value: {first_val})")
                    return modified_output
                return output

            return hook

        hooks = []
        if inhibit_neurons and len(inhibit_neurons) > 0:
            print(f"\n🔧 Registering {len(inhibit_neurons)} inhibition hooks:")
            for layer, neuron in inhibit_neurons:
                if 0 <= layer < len(self.model.gpt_neox.layers):
                    try:
                        target_layer = self.model.gpt_neox.layers[layer]
                        mlp_act = target_layer.mlp.act
                        hook = mlp_act.register_forward_hook(
                            create_inhibition_hook(layer, neuron)
                        )
                        hooks.append(hook)
                        print(f"   ✓ L{layer} N{neuron}")
                    except Exception as e:
                        print(f"   ❌ L{layer} N{neuron}: {e}")

        capture_hooks = []

        def create_capture_hook(layer_idx):
            def hook(module, input, output):
                if output is not None:
                    last_token_activation = output[0, -1, :].detach().cpu().numpy()
                    layer_activations[layer_idx].append(last_token_activation)

            return hook

        for layer in monitor_layers:
            if 0 <= layer < len(self.model.gpt_neox.layers):
                target_layer = self.model.gpt_neox.layers[layer]
                mlp_act = target_layer.mlp.act
                hook = mlp_act.register_forward_hook(create_capture_hook(layer))
                capture_hooks.append(hook)

        print(f"\n📊 Processing {len(sentences)} sentences with {repeats} repeats each...")

        with torch.no_grad():
            for sent_idx, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                print(f"   Sentence {sent_idx + 1}: {sentence[:50]}...")
                for r in range(repeats):
                    try:
                        inputs = self.tokenizer(
                            sentence,
                            return_tensors="pt",
                            truncation=True,
                            max_length=128
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        outputs = self.model(**inputs)
                    except Exception as e:
                        print(f"      Error: {e}")
                        continue

        for hook in hooks + capture_hooks:
            hook.remove()

        if inhibit_neurons and len(inhibit_neurons) > 0 and monitor_layers:
            first_layer = monitor_layers[0]
            if first_layer in layer_activations and len(layer_activations[first_layer]) > 0:
                sample_act = layer_activations[first_layer][0]
                for layer, neuron in inhibit_neurons:
                    if layer in monitor_layers and neuron < len(sample_act):
                        avg_value = np.mean([act[neuron] for act in layer_activations[layer]])
                        print(f"\n✅ Verification - L{layer} N{neuron}: Avg value = {avg_value:.6f}")
                        if abs(avg_value) < 0.0001:
                            print(f"   ✓ Intervention SUCCESSFUL: Neuron is zero")
                        else:
                            print(f"   ⚠️ Intervention FAILED: Neuron is not zero")

        fingerprint_parts = []
        for layer in sorted(monitor_layers):
            if layer in layer_activations and len(layer_activations[layer]) > 0:
                avg_activation = np.mean(layer_activations[layer], axis=0)
                fingerprint_parts.append(avg_activation)
                print(f"   Layer {layer}: {len(layer_activations[layer])} samples, dim={len(avg_activation)}")
            else:
                print(f"   ⚠️ Layer {layer}: No activations captured")
                fingerprint_parts.append(np.zeros(self.d_mlp))

        if fingerprint_parts:
            fingerprint = np.concatenate(fingerprint_parts)
            print(f"\n✅ Generated fingerprint with dimension: {len(fingerprint)}")
            return fingerprint

        return np.array([])


# ==================== MODULE 1: MONITOR SETTINGS ====================
class MonitorModule:
    """Module 1: Global monitor layer settings"""

    def __init__(self, model_manager):
        self.n_layers = model_manager.n_layers
        self.d_mlp = model_manager.d_mlp

    def create_ui(self):
        """Create monitor settings UI"""
        with gr.Column():
            gr.Markdown("## 🔬 Module 1: Monitor Layer Settings")
            gr.Markdown("**Global Setting** - Select which layers to observe for all fingerprints")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        f"**Model:** Pythia-410M | **Layers:** 0-{self.n_layers - 1} | **Neurons/layer:** {self.d_mlp} | **Total:** {self.n_layers * self.d_mlp:,}")

            with gr.Row():
                with gr.Column(scale=1):
                    monitor_start = gr.Number(
                        label="Start Layer",
                        minimum=0,
                        maximum=self.n_layers - 1,
                        value=20,
                        step=1,
                        precision=0
                    )
                    monitor_end = gr.Number(
                        label="End Layer",
                        minimum=0,
                        maximum=self.n_layers - 1,
                        value=23,
                        step=1,
                        precision=0
                    )
                    add_range_btn = gr.Button("➕ Add Layer Range", variant="secondary")

                with gr.Column(scale=1):
                    monitor_layer = gr.Number(
                        label="Individual Layer",
                        minimum=0,
                        maximum=self.n_layers - 1,
                        value=20,
                        step=1,
                        precision=0
                    )
                    add_layer_btn = gr.Button("➕ Add Single Layer")

            with gr.Row():
                clear_monitor_btn = gr.Button("🗑️ Clear All Monitor Layers", variant="secondary")

            monitor_list = gr.Dataframe(
                label="Currently Monitoring Layers",
                headers=["Layer"],
                interactive=False,
                wrap=True
            )

            monitor_dimensions = gr.Textbox(
                label="Total Fingerprint Dimensions",
                value="0",
                interactive=False
            )

            apply_monitor_btn = gr.Button("✅ Apply Monitor Settings", variant="primary", size="lg")
            monitor_status = gr.Textbox(label="Status", value="No monitor layers set", interactive=False)

            return {
                "monitor_start": monitor_start,
                "monitor_end": monitor_end,
                "add_range_btn": add_range_btn,
                "monitor_layer": monitor_layer,
                "add_layer_btn": add_layer_btn,
                "clear_monitor_btn": clear_monitor_btn,
                "monitor_list": monitor_list,
                "monitor_dimensions": monitor_dimensions,
                "apply_monitor_btn": apply_monitor_btn,
                "monitor_status": monitor_status
            }


# ==================== MODULE 2: INTERVENTION SETTINGS ====================
class InterventionModule:
    """Module 2: Select neurons to inhibit"""

    def __init__(self, model_manager):
        self.d_mlp = model_manager.d_mlp
        self.n_layers = model_manager.n_layers

    def create_ui(self):
        """Create intervention settings UI"""
        with gr.Column():
            gr.Markdown("## 🎯 Module 2: Intervention Settings")
            gr.Markdown("**Select neurons to inhibit during intervention fingerprint generation**")

            with gr.Row():
                with gr.Column(scale=1):
                    inhibit_layer = gr.Number(
                        label="Layer",
                        minimum=0,
                        maximum=self.n_layers - 1,
                        value=10,
                        step=1,
                        precision=0
                    )
                with gr.Column(scale=1):
                    inhibit_neuron = gr.Number(
                        label="Neuron Index",
                        minimum=0,
                        maximum=self.d_mlp - 1,
                        value=2000,
                        step=1,
                        precision=0
                    )

            with gr.Row():
                add_inhibit_btn = gr.Button("➕ Add to Inhibition List", variant="primary")
                remove_inhibit_btn = gr.Button("➖ Remove Selected", variant="secondary")
                clear_inhibit_btn = gr.Button("🗑️ Clear All", variant="secondary")

            inhibit_list = gr.Dataframe(
                label="Neurons to Inhibit",
                headers=["Layer", "Neuron", "ID"],
                interactive=False,
                wrap=True
            )

            inhibit_count = gr.Textbox(
                label="Inhibition Count",
                value="0 neurons selected",
                interactive=False
            )

            selected_index = gr.Number(
                label="Row Index to Remove (0-based)",
                value=0,
                minimum=0,
                step=1,
                precision=0
            )

            return {
                "inhibit_layer": inhibit_layer,
                "inhibit_neuron": inhibit_neuron,
                "add_inhibit_btn": add_inhibit_btn,
                "remove_inhibit_btn": remove_inhibit_btn,
                "clear_inhibit_btn": clear_inhibit_btn,
                "inhibit_list": inhibit_list,
                "inhibit_count": inhibit_count,
                "selected_index": selected_index
            }


# ==================== MODULE 3: FINGERPRINT GENERATION ====================
class FingerprintModule:
    """Module 3: Input text, set repeats, generate fingerprints"""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def generate_and_save(self,
                          name: str,
                          fp_type: str,
                          text: str,
                          repeats: int,
                          monitor_layers: List[int],
                          inhibit_neurons: List[Tuple[int, int]]) -> Dict:
        """Generate and save fingerprint"""
        try:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
            if not sentences:
                return {"success": False, "error": "No sentences provided"}

            if not monitor_layers:
                return {"success": False, "error": "No monitor layers set"}

            print(f"\n{'=' * 50}")
            print(f"🔬 Generating {fp_type} fingerprint")
            print(f"   Name: {name}")
            print(f"   Sentences: {len(sentences)}")
            print(f"   Repeats: {repeats}")
            print(f"   Monitor layers: {monitor_layers}")

            if fp_type == "Intervention":
                print(f"   Inhibit neurons: {inhibit_neurons}")
            else:
                print(f"   Inhibit neurons: None (Baseline)")
            print(f"{'=' * 50}\n")

            inhibit = inhibit_neurons if fp_type == "Intervention" and inhibit_neurons else None
            fingerprint = self.model_manager.get_fingerprint(
                sentences, monitor_layers, inhibit, repeats
            )

            if len(fingerprint) == 0:
                return {"success": False, "error": "Fingerprint generation failed - no activations captured"}

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_name = name.replace(" ", "_").replace("/", "_")
            filename = f"fingerprint_{fp_type.lower()}_{clean_name}_{timestamp}.csv"

            save_dir = Path("fingerprints_410m")
            save_dir.mkdir(exist_ok=True)
            filepath = save_dir / filename

            df = pd.DataFrame({"dimension": range(len(fingerprint)), "value": fingerprint})
            df.to_csv(filepath, index=False)

            metadata = {
                "model": "Pythia-410M",
                "name": name,
                "type": fp_type,
                "timestamp": timestamp,
                "num_sentences": len(sentences),
                "repeats": repeats,
                "monitor_layers": monitor_layers,
                "inhibit_neurons": inhibit_neurons if fp_type == "Intervention" else [],
                "vector_dim": len(fingerprint),
                "sentences": sentences[:3]
            }

            meta_path = save_dir / f"metadata_{fp_type.lower()}_{clean_name}_{timestamp}.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            if fp_type == "Intervention" and inhibit_neurons and monitor_layers:
                for layer, neuron in inhibit_neurons:
                    if layer in monitor_layers:
                        layer_idx = monitor_layers.index(layer)
                        offset = layer_idx * self.model_manager.d_mlp + neuron
                        if offset < len(fingerprint):
                            value = fingerprint[offset]
                            print(f"\n🔍 VERIFICATION: L{layer} N{neuron} value = {value:.6f}")
                            if abs(value) < 0.0001:
                                print(f"   ✅ Intervention SUCCESSFUL - neuron is zero")
                                metadata["intervention_verified"] = True
                                metadata["neuron_value"] = float(value)
                            else:
                                print(f"   ❌ Intervention FAILED - neuron is {value:.6f}, should be 0")
                                metadata["intervention_verified"] = False
                                metadata["neuron_value"] = float(value)

            return {
                "success": True,
                "filepath": str(filepath),
                "filename": filename,
                "dimensions": len(fingerprint),
                "metadata": metadata,
                "message": f"✅ Generated {fp_type} fingerprint: {filename} ({len(fingerprint)} dims)"
            }

        except Exception as e:
            print(f"❌ Error generating fingerprint: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def create_ui(self):
        """Create fingerprint generation UI"""
        with gr.Column():
            gr.Markdown("## 📝 Module 3: Fingerprint Generation")
            gr.Markdown("**Input text, set repeats, and generate fingerprints**")

            with gr.Row():
                with gr.Column(scale=2):
                    fingerprint_name = gr.Textbox(
                        label="Fingerprint Name",
                        placeholder="e.g., structured_requests, control_sentences",
                        value="experiment"
                    )

                    text_input = gr.Textbox(
                        label="Sentences (one per line)",
                        placeholder="Enter your sentences here, one per line...\n\nExample:\nTeach me Python programming.\nExplain how neural networks work.\nHow to bake a cake?",
                        lines=6
                    )

                    with gr.Row():
                        repeats_slider = gr.Slider(
                            label="Repeats per Sentence (for average)",
                            minimum=1,
                            maximum=10,
                            value=2,
                            step=1
                        )

                    with gr.Row():
                        generate_baseline_btn = gr.Button("🔵 Generate Baseline Fingerprint (No Inhibition)",
                                                          variant="secondary", size="lg")
                        generate_intervention_btn = gr.Button("🔴 Generate Intervention Fingerprint (With Inhibition)",
                                                              variant="primary", size="lg")

                    fingerprint_status = gr.Textbox(label="Generation Status", lines=3)

                with gr.Column(scale=1):
                    gr.Markdown("### Current Settings")
                    current_monitor = gr.Textbox(
                        label="Monitor Layers",
                        value="Not set",
                        interactive=False
                    )
                    current_inhibit = gr.Textbox(
                        label="Active Inhibitions",
                        value="None",
                        interactive=False
                    )

            with gr.Row():
                fingerprint_preview = gr.JSON(label="Fingerprint Info", visible=True)

            with gr.Row():
                gr.Markdown("### 💾 Saved Fingerprints")
                refresh_files_btn = gr.Button("🔄 Refresh File List")

                # Changed from Dropdown to File Explorer
                saved_files = gr.File(
                    label="Click to Browse Saved Fingerprints",
                    file_count="multiple",
                    file_types=[".csv"],
                    interactive=True
                )
                file_info = gr.JSON(label="File Details", visible=False)

            return {
                "fingerprint_name": fingerprint_name,
                "text_input": text_input,
                "repeats_slider": repeats_slider,
                "generate_baseline_btn": generate_baseline_btn,
                "generate_intervention_btn": generate_intervention_btn,
                "fingerprint_status": fingerprint_status,
                "fingerprint_preview": fingerprint_preview,
                "saved_files": saved_files,
                "refresh_files_btn": refresh_files_btn,
                "file_info": file_info,
                "current_monitor": current_monitor,
                "current_inhibit": current_inhibit
            }


# ==================== MODULE 4: COMPARISON ====================
class ComparisonModule:
    """Module 4: Compare baseline vs intervention fingerprints"""

    def __init__(self):
        pass

    def load_fingerprint(self, filepath: str) -> Optional[np.ndarray]:
        """Load fingerprint from CSV"""
        if not filepath or not os.path.exists(filepath):
            return None
        try:
            df = pd.read_csv(filepath)
            if 'value' in df.columns:
                return df['value'].values.astype(np.float32)
            return None
        except:
            return None

    def calculate_similarity(self, fp1_path: str, fp2_path: str) -> Dict:
        """Calculate cosine similarity between two fingerprints"""
        try:
            fp1 = self.load_fingerprint(fp1_path)
            fp2 = self.load_fingerprint(fp2_path)

            if fp1 is None or fp2 is None:
                return {"success": False, "error": "Failed to load files"}

            if len(fp1) != len(fp2):
                return {"success": False, "error": f"Dimension mismatch: {len(fp1)} vs {len(fp2)}"}

            similarity = 1 - cosine(fp1, fp2)

            meta1_path = fp1_path.replace("fingerprint_", "metadata_").replace(".csv", ".json")
            meta2_path = fp2_path.replace("fingerprint_", "metadata_").replace(".csv", ".json")

            metadata1 = {}
            metadata2 = {}

            if os.path.exists(meta1_path):
                with open(meta1_path, 'r', encoding='utf-8') as f:
                    metadata1 = json.load(f)
            if os.path.exists(meta2_path):
                with open(meta2_path, 'r', encoding='utf-8') as f:
                    metadata2 = json.load(f)

            return {
                "success": True,
                "similarity": float(similarity),
                "drop": 1 - float(similarity),
                "fp1_name": os.path.basename(fp1_path),
                "fp2_name": os.path.basename(fp2_path),
                "fp1_info": metadata1,
                "fp2_info": metadata2,
                "dimensions": len(fp1)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_ui(self):
        """Create comparison module UI - Using File Explorer instead of Dropdown"""
        with gr.Column():
            gr.Markdown("## 🔍 Module 4: Fingerprint Comparison")
            gr.Markdown("**Compare baseline vs intervention fingerprints**")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Baseline Fingerprint")
                    # Changed from Dropdown to File
                    fp_baseline = gr.File(
                        label="Select Baseline CSV File",
                        file_types=[".csv"],
                        interactive=True
                    )
                    baseline_info = gr.JSON(label="Baseline Info", visible=False)

                with gr.Column(scale=1):
                    gr.Markdown("### Intervention Fingerprint")
                    # Changed from Dropdown to File
                    fp_intervention = gr.File(
                        label="Select Intervention CSV File",
                        file_types=[".csv"],
                        interactive=True
                    )
                    intervention_info = gr.JSON(label="Intervention Info", visible=False)

            with gr.Row():
                compare_btn = gr.Button("📊 Calculate Cosine Similarity", variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=1):
                    similarity_output = gr.Number(
                        label="Cosine Similarity",
                        value=0.0,
                        precision=4,
                        interactive=False
                    )
                    similarity_drop = gr.Number(
                        label="Similarity Drop (1 - similarity)",
                        value=0.0,
                        precision=4,
                        interactive=False
                    )

                with gr.Column(scale=2):
                    similarity_gauge = gr.Plot(label="Similarity Gauge")

            with gr.Row():
                comparison_details = gr.JSON(label="Comparison Details", visible=True)

            with gr.Row():
                gr.Markdown("### 📁 Select CSV files from your computer")

            return {
                "fp_baseline": fp_baseline,
                "fp_intervention": fp_intervention,
                "baseline_info": baseline_info,
                "intervention_info": intervention_info,
                "compare_btn": compare_btn,
                "similarity_output": similarity_output,
                "similarity_drop": similarity_drop,
                "similarity_gauge": similarity_gauge,
                "comparison_details": comparison_details
            }


# ==================== Main Application ====================
class NeuronInterventionApp:
    """Main application - Four module structure with working intervention for Pythia-410M"""

    def __init__(self, model_path: str):
        print("\n" + "=" * 60)
        print("🧠 NEURON INTERVENTION EXPERIMENT SYSTEM (Pythia-410M)")
        print("=" * 60)
        print("Four Module Structure:")
        print("  Module 1: Set Monitor Layers (Global)")
        print("  Module 2: Set Intervention Neurons")
        print("  Module 3: Input Text + Repeats + Generate Fingerprints")
        print("  Module 4: Compare Baseline vs Intervention")
        print("=" * 60)
        print("\n✅ Intervention FIXED: Now uses tensor cloning")
        print(f"📊 Model: 24 layers, 4096 neurons/layer, Total: 98,304 neurons")
        print("=" * 60)

        self.model_manager = ModelManager(model_path)
        self.monitor_module = MonitorModule(self.model_manager)
        self.intervention_module = InterventionModule(self.model_manager)
        self.fingerprint_module = FingerprintModule(self.model_manager)
        self.comparison_module = ComparisonModule()

        self.current_monitor_layers = []
        self.current_inhibit_neurons = []

    def get_fingerprint_files(self):
        """Get list of saved fingerprint files (kept for reference but not used in UI)"""
        fingerprint_dir = Path("fingerprints_410m")
        if not fingerprint_dir.exists():
            fingerprint_dir.mkdir(exist_ok=True)
        return list(fingerprint_dir.glob("fingerprint_*.csv"))

    def create_interface(self):
        """Create main interface"""
        with gr.Blocks(title="🧠 Neuron Intervention System (Pythia-410M)", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🧠 Neuron Intervention Experiment System (Pythia-410M)")
            gr.Markdown("### Four-Module Structure: Monitor → Intervene → Generate → Compare")
            gr.Markdown("### ✅ Intervention Fixed: Uses tensor cloning to ensure neurons are set to zero")
            gr.Markdown("### 📊 24 Layers | 4096 Neurons/Layer | Total: 98,304 Neurons")

            # ========== MODULE 1: MONITOR SETTINGS ==========
            gr.Markdown("---")
            monitor_ui = self.monitor_module.create_ui()

            monitor_df_state = gr.State([])

            def add_monitor_range(start, end, current):
                current = current if current else []
                for layer in range(int(start), int(end) + 1):
                    if [layer] not in current:
                        current.append([layer])
                df = pd.DataFrame(current, columns=["Layer"])
                dims = len(current) * self.model_manager.d_mlp
                dims_text = f"{dims} dimensions ({len(current)} layers × {self.model_manager.d_mlp} neurons)"
                return df, current, dims_text

            monitor_ui["add_range_btn"].click(
                fn=add_monitor_range,
                inputs=[monitor_ui["monitor_start"], monitor_ui["monitor_end"], monitor_df_state],
                outputs=[monitor_ui["monitor_list"], monitor_df_state, monitor_ui["monitor_dimensions"]]
            )

            def add_monitor_layer(layer, current):
                current = current if current else []
                if [layer] not in current:
                    current.append([layer])
                df = pd.DataFrame(current, columns=["Layer"])
                dims = len(current) * self.model_manager.d_mlp
                dims_text = f"{dims} dimensions ({len(current)} layers × {self.model_manager.d_mlp} neurons)"
                return df, current, dims_text

            monitor_ui["add_layer_btn"].click(
                fn=add_monitor_layer,
                inputs=[monitor_ui["monitor_layer"], monitor_df_state],
                outputs=[monitor_ui["monitor_list"], monitor_df_state, monitor_ui["monitor_dimensions"]]
            )

            def clear_monitor():
                return pd.DataFrame(columns=["Layer"]), [], "0"

            monitor_ui["clear_monitor_btn"].click(
                fn=clear_monitor,
                outputs=[monitor_ui["monitor_list"], monitor_df_state, monitor_ui["monitor_dimensions"]]
            )

            def apply_monitor_settings(monitor_df):
                layers = []
                if monitor_df:
                    for row in monitor_df:
                        if len(row) > 0:
                            layers.append(int(row[0]))

                self.current_monitor_layers = layers
                if layers:
                    status = f"✅ Monitoring layers: {sorted(layers)}"
                    monitor_text = f"Layers: {sorted(layers)}"
                else:
                    status = "⚠️ No monitor layers set"
                    monitor_text = "Not set"

                return status, monitor_text

            monitor_text_state = gr.State("Not set")

            monitor_ui["apply_monitor_btn"].click(
                fn=apply_monitor_settings,
                inputs=[monitor_df_state],
                outputs=[monitor_ui["monitor_status"], monitor_text_state]
            )

            # ========== MODULE 2: INTERVENTION SETTINGS ==========
            gr.Markdown("---")
            intervention_ui = self.intervention_module.create_ui()

            inhibit_df_state = gr.State([])
            inhibit_text_state = gr.State("None")

            def add_inhibit(layer, neuron, current):
                current = current if current else []

                if neuron < 0 or neuron >= self.model_manager.d_mlp:
                    return gr.DataFrame(), current, f"❌ Neuron must be 0-{self.model_manager.d_mlp - 1}", "None"

                exists = False
                for item in current:
                    if len(item) >= 2 and item[0] == layer and item[1] == neuron:
                        exists = True
                        break

                if not exists:
                    current.append([int(layer), int(neuron), f"L{int(layer)}_N{int(neuron)}"])

                if current:
                    df = pd.DataFrame(current, columns=["Layer", "Neuron", "ID"])
                else:
                    df = pd.DataFrame(columns=["Layer", "Neuron", "ID"])

                self.current_inhibit_neurons = [(int(row[0]), int(row[1])) for row in current if len(row) >= 2]

                if current:
                    neurons = [f"L{row[0]}N{row[1]}" for row in current if len(row) >= 2]
                    inhibit_text = f"{len(neurons)} neurons: {', '.join(neurons[:3])}"
                    if len(neurons) > 3:
                        inhibit_text += f" (+{len(neurons) - 3} more)"
                else:
                    inhibit_text = "None"

                return df, current, f"{len(current)} neurons selected", inhibit_text

            intervention_ui["add_inhibit_btn"].click(
                fn=add_inhibit,
                inputs=[intervention_ui["inhibit_layer"], intervention_ui["inhibit_neuron"], inhibit_df_state],
                outputs=[intervention_ui["inhibit_list"], inhibit_df_state,
                         intervention_ui["inhibit_count"], inhibit_text_state]
            )

            def remove_inhibit(index, current):
                current = current if current else []
                if 0 <= index < len(current):
                    removed = current.pop(index)
                    print(f"Removed: {removed}")

                if current:
                    df = pd.DataFrame(current, columns=["Layer", "Neuron", "ID"])
                else:
                    df = pd.DataFrame(columns=["Layer", "Neuron", "ID"])

                self.current_inhibit_neurons = [(int(row[0]), int(row[1])) for row in current if len(row) >= 2]

                if current:
                    neurons = [f"L{row[0]}N{row[1]}" for row in current if len(row) >= 2]
                    inhibit_text = f"{len(neurons)} neurons: {', '.join(neurons[:3])}"
                    if len(neurons) > 3:
                        inhibit_text += f" (+{len(neurons) - 3} more)"
                else:
                    inhibit_text = "None"

                return df, current, f"{len(current)} neurons selected", inhibit_text

            intervention_ui["remove_inhibit_btn"].click(
                fn=remove_inhibit,
                inputs=[intervention_ui["selected_index"], inhibit_df_state],
                outputs=[intervention_ui["inhibit_list"], inhibit_df_state,
                         intervention_ui["inhibit_count"], inhibit_text_state]
            )

            def clear_inhibit():
                self.current_inhibit_neurons = []
                df = pd.DataFrame(columns=["Layer", "Neuron", "ID"])
                return df, [], "0 neurons selected", "None"

            intervention_ui["clear_inhibit_btn"].click(
                fn=clear_inhibit,
                outputs=[intervention_ui["inhibit_list"], inhibit_df_state,
                         intervention_ui["inhibit_count"], inhibit_text_state]
            )

            # ========== MODULE 3: FINGERPRINT GENERATION ==========
            gr.Markdown("---")
            fingerprint_ui = self.fingerprint_module.create_ui()

            def update_fingerprint_display(monitor_text, inhibit_text):
                return monitor_text, inhibit_text

            monitor_text_state.change(
                fn=update_fingerprint_display,
                inputs=[monitor_text_state, inhibit_text_state],
                outputs=[fingerprint_ui["current_monitor"], fingerprint_ui["current_inhibit"]]
            )

            inhibit_text_state.change(
                fn=update_fingerprint_display,
                inputs=[monitor_text_state, inhibit_text_state],
                outputs=[fingerprint_ui["current_monitor"], fingerprint_ui["current_inhibit"]]
            )

            def generate_baseline(name, text, repeats, monitor_df):
                monitor_layers = []
                if monitor_df:
                    for row in monitor_df:
                        if len(row) > 0:
                            monitor_layers.append(int(row[0]))

                result = self.fingerprint_module.generate_and_save(
                    name, "Baseline", text, int(repeats), monitor_layers, []
                )

                if result["success"]:
                    return result, result["message"], result
                else:
                    return result, f"❌ {result['error']}", {}

            def generate_intervention(name, text, repeats, monitor_df, inhibit_df):
                monitor_layers = []
                if monitor_df:
                    for row in monitor_df:
                        if len(row) > 0:
                            monitor_layers.append(int(row[0]))

                inhibit_neurons = []
                if inhibit_df:
                    for row in inhibit_df:
                        if len(row) >= 2:
                            inhibit_neurons.append((int(row[0]), int(row[1])))

                result = self.fingerprint_module.generate_and_save(
                    name, "Intervention", text, int(repeats), monitor_layers, inhibit_neurons
                )

                if result["success"]:
                    return result, result["message"], result
                else:
                    return result, f"❌ {result['error']}", {}

            fingerprint_ui["generate_baseline_btn"].click(
                fn=generate_baseline,
                inputs=[fingerprint_ui["fingerprint_name"], fingerprint_ui["text_input"],
                        fingerprint_ui["repeats_slider"], monitor_df_state],
                outputs=[fingerprint_ui["fingerprint_preview"], fingerprint_ui["fingerprint_status"],
                         fingerprint_ui["file_info"]]
            )

            fingerprint_ui["generate_intervention_btn"].click(
                fn=generate_intervention,
                inputs=[fingerprint_ui["fingerprint_name"], fingerprint_ui["text_input"],
                        fingerprint_ui["repeats_slider"], monitor_df_state, inhibit_df_state],
                outputs=[fingerprint_ui["fingerprint_preview"], fingerprint_ui["fingerprint_status"],
                         fingerprint_ui["file_info"]]
            )

            # Simplified refresh - just updates the file explorer
            def refresh_files():
                return None  # File explorer doesn't need refresh

            fingerprint_ui["refresh_files_btn"].click(
                fn=refresh_files,
                outputs=[fingerprint_ui["saved_files"]]
            )

            def load_file_info(filepath):
                if not filepath:
                    return {}
                # Handle both string and list inputs
                if isinstance(filepath, list):
                    if not filepath:
                        return {}
                    filepath = filepath[0]

                meta_path = str(filepath).replace("fingerprint_", "metadata_").replace(".csv", ".json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                return {"file": os.path.basename(str(filepath))}

            fingerprint_ui["saved_files"].change(
                fn=load_file_info,
                inputs=[fingerprint_ui["saved_files"]],
                outputs=[fingerprint_ui["file_info"]]
            )

            # ========== MODULE 4: COMPARISON ==========
            gr.Markdown("---")
            comparison_ui = self.comparison_module.create_ui()

            # Load file info for comparison
            def load_baseline_info(file):
                if not file:
                    return {}
                filepath = file.name if hasattr(file, 'name') else file
                meta_path = str(filepath).replace("fingerprint_", "metadata_").replace(".csv", ".json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                return {"file": os.path.basename(str(filepath))}

            def load_intervention_info(file):
                if not file:
                    return {}
                filepath = file.name if hasattr(file, 'name') else file
                meta_path = str(filepath).replace("fingerprint_", "metadata_").replace(".csv", ".json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                return {"file": os.path.basename(str(filepath))}

            comparison_ui["fp_baseline"].change(
                fn=load_baseline_info,
                inputs=[comparison_ui["fp_baseline"]],
                outputs=[comparison_ui["baseline_info"]]
            )

            comparison_ui["fp_intervention"].change(
                fn=load_intervention_info,
                inputs=[comparison_ui["fp_intervention"]],
                outputs=[comparison_ui["intervention_info"]]
            )

            def calculate_similarity(fp1, fp2):
                if not fp1 or not fp2:
                    return 0, 0, None, {"error": "Please select both fingerprint files"}

                # Extract file paths from File objects
                fp1_path = fp1.name if hasattr(fp1, 'name') else fp1
                fp2_path = fp2.name if hasattr(fp2, 'name') else fp2

                result = self.comparison_module.calculate_similarity(fp1_path, fp2_path)

                if result["success"]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["similarity"],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Cosine Similarity"},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightcoral"},
                                {'range': [0.3, 0.7], 'color': "lightyellow"},
                                {'range': [0.7, 1], 'color': "lightgreen"}
                            ]
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))

                    return result["similarity"], result["drop"], fig, result
                else:
                    return 0, 0, None, result

            comparison_ui["compare_btn"].click(
                fn=calculate_similarity,
                inputs=[comparison_ui["fp_baseline"], comparison_ui["fp_intervention"]],
                outputs=[comparison_ui["similarity_output"], comparison_ui["similarity_drop"],
                         comparison_ui["similarity_gauge"], comparison_ui["comparison_details"]]
            )

            # Initialize
            app.load(
                fn=clear_inhibit,
                outputs=[intervention_ui["inhibit_list"], inhibit_df_state,
                         intervention_ui["inhibit_count"], inhibit_text_state]
            )

        return app


# ==================== Main ====================
def main():
    MODEL_PATH = r"E:\AI_Models\Pythia\Pythia-410M\models"

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model path not found: {MODEL_PATH}")
        print("Please check the path and try again.")
        return

    try:
        app = NeuronInterventionApp(MODEL_PATH)
        interface = app.create_interface()

        print("\n🌐 Starting Gradio interface...")
        print("🔗 Local URL: http://127.0.0.1:7864")
        print("📁 Fingerprints saved to: ./fingerprints_410m/")
        print("\n" + "=" * 60)
        print("✅ System ready! Intervention is now WORKING:")
        print("   - Uses tensor .clone() to modify activations")
        print("   - Prints verification of neuron values")
        print("   - You should see neurons being set to 0")
        print("=" * 60 + "\n")

        interface.launch(
            server_name="127.0.0.1",
            server_port=7864,
            share=False,
            quiet=False
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()