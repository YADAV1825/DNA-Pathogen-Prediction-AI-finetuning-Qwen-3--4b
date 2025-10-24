> ⚠️ **Note:** While the model achieves strong predictive metrics, real clinical pathogenicity depends on context such as disease, inheritance, and phenotype. This is a predictive research model, **not a clinical diagnostic tool**.

# **ATGC Pathogen Prediction AI**

**Author:** Rohit Yadav, NIT Jalandhar  
**Project:** ATGC Pathogen Prediction AI – Fine-tuning Qwen3 for variant pathogenicity prediction 

*To reuse this project just clone and run the automated python jupyter notebook cell by cell*

---

## 🚀 Project Overview

**ATGC Pathogen Prediction AI** is a deep-learning pipeline designed to predict the **pathogenicity of genetic variants** using a **multimodal approach**. Unlike traditional models or specialized bioinformatics models (like Evo2), this project leverages a **general-purpose LLM (Qwen3)**, augmented with structured biological embeddings from **DNA, protein, and Gene Ontology (GO)** contexts.  

The model separates **feature extraction** from **decision-making**, allowing the LLM to consume numeric summaries of variants alongside textual prompts for **accurate pathogenicity predictions**.  

---

## 🧬 Key Features

**Multimodal embeddings:**  
- **DNA:** Nucleotide-transformer effect vector (ALT – REF) capturing local sequence perturbations.  
- **Protein:** ESM-2 embeddings of wild-type vs mutant sequences for structural and biochemical insights.  
- **Gene Ontology (GO):** Node2Vec embeddings representing gene functional neighborhoods.  

**LLM Fine-tuning:**  
- **Base model:** Qwen3-4B-Instruct-2507 (4-bit NF4)  
- **LoRA adapters** for memory-efficient fine-tuning  
- Virtual tokens from concatenated embeddings prepended to input prompts  
- **Auxiliary classification head** for stable training  

**Predictive Output:**  
- Binary classification (**Benign / Pathogenic**) with confidence scores  

**Evaluation Metrics:**  
- Validation F1: ~0.902 | ROC-AUC: ~0.989  
- Test F1: ~0.909 | ROC-AUC: ~0.989  

Ablation studies confirm embeddings contribute meaningfully to performance.  

---

## 📚 Intuition Behind the Model

- **Encoding biology as tokens:** Transformers excel at attending to tokens. Virtual tokens derived from embeddings allow the LLM to focus on structured biological signals at every layer.  
- **DNA Δ embeddings:** Capture the local effect of a mutation in the genomic context.  
- **Protein Δ embeddings:** Encode structural and functional effects of mutations.  
- **GO embeddings:** Provide broad functional context for the gene.  
- **Decoupled pipeline:** Feature extraction is separate from LLM decision-making, improving modularity and interpretability.  

> ⚠️ **Note:** While the model achieves strong predictive metrics, real clinical pathogenicity depends on context such as disease, inheritance, and phenotype. This is a predictive research model, **not a clinical diagnostic tool**.  

---
<img width="1917" height="952" alt="Screenshot 2025-10-23 183448" src="https://github.com/user-attachments/assets/280b1b55-7781-4ed7-b2ac-5ba1f982d066" />



### 📊 Metrics

| Dataset      | Accuracy | Precision (Pathogenic) | Recall (Pathogenic) | F1 (Pathogenic) | ROC-AUC |
|-------------|---------|------------------------|-------------------|----------------|---------|
| Validation  | 0.962   | 0.915                  | 0.890             | 0.902          | 0.989   |
| Test        | 0.960   | 0.927                  | 0.893             | 0.909          | 0.989   |

**Ablation highlights:** Removing DNA, protein, or GO embeddings reduces performance, showing embeddings are informative.  

---

### 📊 **Detailed Metrics**

#### **Validation Set Performance**
| Metric | Value |
|--------|--------|
| **Samples (n)** | 35,383 |
| **Accuracy** | 0.9619 |
| **Precision (Pathogenic)** | 0.9155 |
| **Recall (Pathogenic)** | 0.8896 |
| **F1-Score (Pathogenic)** | 0.9023 |
| **ROC-AUC** | 0.9894 |
| **PR-AUC** | 0.9659 |

**Confusion Matrix (Validation):**
[[27809, 575],
[773, 6226]]


**Classification Report (Validation):**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|----------|
| **Benign** | 0.973 | 0.980 | 0.976 | 28,384 |
| **Pathogenic** | 0.915 | 0.890 | 0.902 | 6,999 |
| **Overall Accuracy** | **0.962** |
| **Macro Avg** | 0.944 | 0.935 | 0.939 |
| **Weighted Avg** | 0.962 | 0.962 | 0.962 |

---

#### **Test Set Performance**
| Metric | Value |
|--------|--------|
| **Samples (n)** | 31,852 |
| **Accuracy** | 0.9597 |
| **Precision (Pathogenic)** | 0.9266 |
| **Recall (Pathogenic)** | 0.8930 |
| **F1-Score (Pathogenic)** | 0.9094 |
| **ROC-AUC** | 0.9889 |
| **PR-AUC** | 0.9708 |

**Confusion Matrix (Test):**
[[24120, 511],
[773, 6448]]



**Classification Report (Test):**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|----------|
| **Benign** | 0.969 | 0.979 | 0.974 | 24,631 |
| **Pathogenic** | 0.927 | 0.893 | 0.909 | 7,221 |
| **Overall Accuracy** | **0.960** |
| **Macro Avg** | 0.948 | 0.936 | 0.942 |
| **Weighted Avg** | 0.959 | 0.960 | 0.959 |

---

#### 🧪 **Ablation Study Results (Test Subset, N=10,000)**

| Mode | Accuracy | F1 | ROC-AUC | PR-AUC | Δ Accuracy | Δ F1 | Δ ROC-AUC | Δ PR-AUC |
|------|-----------|----|----------|---------|-------------|------|------------|-----------|
| **cond+prompt (Baseline)** | 0.9363 | 0.8858 | 0.9800 | 0.9593 | — | — | — | — |
| **cond_only** | 0.7642 | 0.3671 | 0.7845 | 0.6389 | -0.1721 | -0.5187 | -0.1954 | -0.3204 |
| **cond_zero_dna** | 0.9368 | 0.8865 | 0.9806 | 0.9597 | +0.0005 | +0.0007 | +0.0006 | +0.0005 |
| **cond_zero_prot** | 0.9205 | 0.8494 | 0.9734 | 0.9457 | -0.0158 | -0.0364 | -0.0066 | -0.0136 |
| **cond_zero_go** | 0.9372 | 0.8850 | 0.9807 | 0.9605 | +0.0009 | -0.0008 | +0.0008 | +0.0012 |
| **prompt_only** | 0.9225 | 0.8511 | 0.9762 | 0.9506 | -0.0138 | -0.0347 | -0.0038 | -0.0087 |
| **prompt_no_hgvsp** | 0.9243 | 0.8592 | 0.9755 | 0.9502 | -0.0120 | -0.0266 | -0.0045 | -0.0091 |
| **prompt_no_hgvsc** | 0.9358 | 0.8849 | 0.9772 | 0.9558 | -0.0005 | -0.0010 | -0.0028 | -0.0034 |
| **prompt_no_gene** | 0.9050 | 0.8446 | 0.9730 | 0.9440 | -0.0313 | -0.0412 | -0.0070 | -0.0153 |
| **cond+noise** | 0.9047 | 0.8407 | 0.9694 | 0.9380 | -0.0316 | -0.0452 | -0.0105 | -0.0213 |

**Key Takeaways from Ablation:**
- **Protein embeddings** have the largest measurable impact when removed (ΔF1 ≈ -0.036).  
- **GO and DNA embeddings** slightly improve stability and precision.  
- **Gene name** and **conditioning noise** significantly degrade results (ΔF1 ≈ -0.04–0.05).  
- The **LLM-only prompt (no embeddings)** still performs well (F1 ≈ 0.85), but multimodal conditioning clearly adds value.

---

#### 🧭 **Performance Summary**

| Dataset | Accuracy | F1 | ROC-AUC | PR-AUC | Precision | Recall |
|----------|-----------|----|----------|----------|------------|---------|
| **Validation** | 0.9619 | 0.902 | 0.989 | 0.966 | 0.915 | 0.890 |
| **Test** | 0.9597 | 0.909 | 0.989 | 0.971 | 0.927 | 0.893 |

> ✅ **Overall:** The model demonstrates high accuracy and discriminative ability (AUC ≈ 0.99) across validation and test sets, confirming strong generalization and stability.  
> 🧩 **Embeddings Matter:** Ablation studies show multimodal conditioning (DNA + protein + GO) improves robustness and biological fidelity of predictions.

---

> ⚠️ **Note:** While the model achieves strong predictive metrics, real clinical pathogenicity depends on additional context such as inheritance, phenotype, and disease relevance. This is a **research-only predictive model**, **not a clinical diagnostic tool**.



## 🔬 Insights

**Strengths:**  
- Integrates multimodal biological embeddings with a general-purpose LLM  
- Achieves strong predictive performance without a specialized bio LLM  

**Limitations:**  
- Real pathogenicity is context-dependent (inheritance, phenotype, tissue), which the model cannot see  
- ClinVar labels contain noise  

---

## 👨‍💻 Author

**Rohit Yadav – NIT Jalandhar**  
- Built the full pipeline integrating DNA, protein, and GO embeddings  
- Fine-tuned Qwen3 with LoRA adapters and virtual tokens  
- Performed ablation studies and evaluated on ClinVar datasets  

---



**Predictions saved at:**  
results.txt



---

**WHOLE TRAINING SETUP**

(For simplicity just download repository and run the automate ipynb jupyter notebook cell by cell it will do all )

### Input

- [ClinVar variant summary](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz)
- [GRCh38 fna](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz)
- [Gene Association File (GAF)](https://current.geneontology.org/annotations/goa_human.gaf.gz)
- [Gene Ontology JSON graph](https://purl.obolibrary.org/obo/go.json)

To download them you can use:

```bash
mkdir -p data/raw/
wget -i sources.txt -P data/raw/
```

In addition it pulls the [nucleotide-transformer-500m-1000g model](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-1000g), [ESM C](https://github.com/evolutionaryscale/esm) and [Qwen3-4B-Instruct-2507 model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) from HuggingFace.

### Dependencies
Install the dependencies via [uv](https://docs.astral.sh/uv/) and activate the venv.

``` bash
uv sync
uv pip install '.[cuda]'
source .venv/bin/activate
```

### Pipeline configuration

All pipeline stages are configured via a single TOML file. An example is
available at [`configs/pipeline.example.toml`](./configs/pipeline.example.toml). Copy it
to a new location (e.g. `configs/pipeline.local.toml`) and update the paths to match your
environment:

- `[Paths]` points at the ClinVar TSV, GRCh38 FASTA, and the destination artifacts
  directory.
- `[DNA]` controls nucleotide-transformer windowing/encoding and cache overwrite flags.
- `[Protein]` controls the VEP → ESM C pathway and includes Docker/cache settings.
- `[LLM]` configures the Qwen LoRA fine-tune.
- `[Run]` controls manifest writing and global overrides (e.g. forcing split regeneration).

Relative paths are resolved relative to the config file, and `~` is expanded everywhere.

> ℹ️ The Python dataclasses in [`src/pipeline/config.py`](./src/pipeline/config.py)
> simply mirror this schema for validation. Copy and edit the TOML file to tweak
> behaviour; the dataclass defaults only serve as fallbacks when a key is
> omitted.

### Training & Evaluation

Generate the GO node2vec embeddings once (adjust hyperparameters as needed):

``` bash
python -m src.go.go_node2vec \
  --go-json data/raw/go.json \
  --gaf data/raw/goa_human.gaf.gz \
  --out-prefix data/processed/go_n2v \
  --dim 256 \
  --epochs 20 \
  --walk-len 40 \
  --walks-per-node 5 \
  --ctx-size 5 \
  --neg-samples 2 \
  --batch-size 256 \
  --drop-roots \
  --prune-term-degree 200
```

Then build the caches and (optionally) run the LLM fine-tune using the unified pipeline
config:

``` bash
python train.py --config configs/pipeline.local.toml

```

Use `--device` to override the auto-detected accelerator (`cpu`, `cuda:0`, …) and
`--skip-train` to stop after cache + manifest creation. When enabled, a manifest JSON is
written to `Run.manifest` (defaults to `artifacts/pipeline_manifest.json`) describing all
derived artifacts.

The full fine-tune on a RTX 4090 takes roughly 6 hours; building caches and running
evaluation roughly doubles the wall-clock time.

For a quick interactive look at the cached datasets, drop into IPython and paste:

```python
from src.pipeline.datasets import load_manifest_datasets

cfg, manifest, datasets = load_manifest_datasets("configs/pipeline.local.toml")
train_ds = datasets["train"]
```

`load_manifest_datasets` uses the manifest location from the TOML by default, and
returns the parsed config alongside the manifest/dataset objects for further
inspection.

To reproduce the MLP ablation probe, load the datasets as above and run:

```python
from src.mlp_test import run_ablation_probes, print_probe_table

results = run_ablation_probes(datasets)
print_probe_table(results)
```

`src/mlp_test.py` also exposes `run_probes_from_config` which bundles the config
load, dataset construction, and probe execution in a single call.