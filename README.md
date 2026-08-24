# SpeechLLM Meets Federated Learning for End-to-End ASR: English and Italian Case Studies

[![arXiv](https://img.shields.io/badge/arXiv-2607.25716-b31b1b.svg)](https://arxiv.org/abs/2607.25716)
[![Conference](https://img.shields.io/badge/FLICS-2026-blue.svg)](https://www.flics-conference.org/editions/flics2026/index.php)
[![Conference](https://img.shields.io/badge/Interspeech-2026-red.svg)](https://interspeech2026.org/)
[![Framework](https://img.shields.io/badge/FL-Flower-green.svg)](https://flower.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"SpeechLLM Meets Federated Learning for End-to-End ASR: English and Italian Case Studies"**, accepted at FLICS 2026.

**Authors:** Mohamed Nabih Ali, Daniele Falavigna, Alessio Brutti
**Affiliation:** Fondazione Bruno Kessler (FBK), Trento, Italy
**Contact:** mnabih@fbk.eu

---

## 📖 Overview

This repository contains the code for the first systematic study of **federated training for SpeechLLM-based end-to-end ASR**. We propose a communication-efficient federated optimization framework tailored to large speech-language architectures, evaluated on monolingual English and Italian ASR tasks.

**Key contributions:**
- First systematic study of federated learning applied to SpeechLLM-based ASR
- A communication-efficient FL strategy that aggregates only trainable (LoRA + projector) parameters
- A modified FedAvg with unified exponential learning rate decay (**Adaptive FedAvg**)
- Extensive evaluation across two speech encoders (WavLM-Large, Whisper-Medium) and two languages (English, Italian)
- Ablation study comparing full fine-tuning, adapter-based methods, and SpeechLLM/PEFT under FL

<p align="center">
  <img src="assets/architecture.png" alt="Federated SpeechLLM architecture" width="500"/>
</p>

---

## 🏗️ Architecture

The pipeline consists of:
1. **Speech Encoder** — WavLM-Large (317M params) or Whisper-Medium (769M params), frozen
2. **Projector** — two-stage linear adapter (projection + average pooling, k=2) mapping speech embeddings into the 2048-d TinyLlama input space
3. **LLM Backbone** — TinyLlama-1.1B-Chat-v1.0, frozen, adapted via **LoRA**
4. **Federated Training** —Adaptive FedAvg built on Flower-based FedAvg, aggregating only LoRA + projector parameters across clients

---

## 📂 Repository Structure

```
├── data_prepare/
|   |── prepare_data_libri_advanced  # prepare LibriSpeech dataset
|   |── prepare_data_mls_advanced    # prepare MLS dataset
|   |── split_csv_mls.py             # Split the prepared CSV file by speaker for MLS
|   └── split_csv_ls.py              # Split the prepared CSV file by speaker for LibriSpeech
├── models/
│   ├── speech_encoder.py   # WavLM / Whisper wrappers
│   ├── projector.py        # Linear adapter + average pooling
│   └── speechllm.py        # Full SpeechLLM (encoder + projector + LoRA-LLM)
│── client.py               # Flower client (local training loop)
├── test.py                 # WER evaluation
├── test.sh                  # Shell scripts for reproducing experiments
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```
git clone https://github.com/mnabihali/Fed-SpeechLLM.git
cd Fed-SpeechLLM
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Core dependencies:** PyTorch, HuggingFace `transformers`, `peft` (LoRA), `flwr` (Flower), `torchaudio`.

---

## 📊 Datasets

| Dataset | Language | Train (hrs) | Train (spks) | Test (hrs) | Test (spks) |
|---|---|---|---|---|---|
| LibriSpeech-100 (train-clean-100 / test-clean) | English | 100 | 251 | 5.4 | 40 |
| MLS Italian | Italian | 247.38 | 65 | 5.27 | 10 |

```
python data/prepare_librispeech.py --output_dir ./data/librispeech
python data/prepare_mls_italian.py --output_dir ./data/mls_it
```

Each speaker is treated as one federated client (316 clients total in the multilingual setting).

---

## 🚀 Usage

### Data preparation
```
prepare_data_libri_advanced.py or prepare_data_ms_advanced.py    # Run this command to prepare .csv files for each dataset --> used for Central training
split_csv_ls.py or split_csv_mls.py  # Run this command to split the generated .csv files by speaker from the previous script for each dataset --> used for Federated training
```

### Centralized baseline
```
Use this repo to generate the Central Training performance: "https://github.com/skit-ai/SpeechLLM"
```

### Federated training
```
python client.py 
```

### Evaluation
```
test.sh
```

Key hyperparameters (Adaptive FedAvg): `η0 = 0.001`, `γ = 0.9` (decay factor), `τ = 10` (decay period, rounds), `T = 100` (total rounds), 30% client participation per round, 10 local epochs.

---

## 📈 Results

### Adaptive FedAvg vs. vanilla FedAvg (WavLM, LibriSpeech)

| Round | FedAvg WER | Adaptive FedAvg WER |
|---|---|---|
| 20 | 19.7% | 9.7% |
| 100 | 7.9% | 6.4% |

Central training reference: **6.1%**

### Monolingual FL — WavLM vs. Whisper (WER % at round 100)

| Encoder | LibriSpeech (FL) | LibriSpeech (Central) | MLS Italian (FL) | MLS Italian (Central) |
|---|---|---|---|---|
| WavLM-Large | 6.4 | 6.1 | 22.6 | 20.1 |
| Whisper-Medium | 6.6 | 6.0 | 18.7 | 17.5 |

### SpeechLLM vs. full fine-tuning vs. adapters (LibriSpeech)

| Training | Model | # Params | WER (%) |
|---|---|---|---|
| Centralized | WavLM-FT | 85.1M | 4.4 |
| Centralized | WavLM EL-adapters | 9.1M | 4.6 |
| Centralized | Speech-LLM | 8.4M | 6.1 |
| Federated | WavLM-FT | 85.1M | ✗ (fails to converge) |
| Federated | WavLM EL-adapters | 9.1M | 6.1 |
| Federated | Speech-LLM | 8.4M | 6.4 |

### Monolingual → Multilingual (all 316 clients, WavLM, round 100)

| Dataset | Federated WER | Central WER | Gap |
|---|---|---|---|
| LibriSpeech (EN) | 16.8% | 6.1% | +10.7 |
| MLS Italian | 19.7% | 18.4% | +1.3 |

---

## 📄 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{ali2026speechllm,
  title     = {SpeechLLM Meets Federated Learning for End-to-End ASR: English and Italian Case Studies},
  author    = {Ali, Mohamed Nabih and Falavigna, Daniele and Brutti, Alessio},
  booktitle = {Proceedings of FLICS 2026},
  year      = {2026},
  eprint    = {2607.25716},
  archivePrefix = {arXiv}
}
```

```bibtex
@inproceedings{ali2026flspeechllm,
  title     = {Fed-SpeechLLM: Federated Learning Speech Language Models for Multilingual ASR},
  author    = {Ali, Mohamed Nabih and Falavigna, Daniele and Brutti, Alessio},
  booktitle = {Proceedings of Interspeech 2026},
  year      = {2026},
}
```

---

## 🙏 Acknowledgements

This work was carried out at the SpeechTek unit, Fondazione Bruno Kessler (FBK), Trento, Italy.

---

## 📬 Contact

For questions or issues, please open a GitHub issue or contact:
- Mohamed Nabih Ali — mnabih@fbk.eu

---

## 📜 License

This project is released under the [MIT License](LICENSE).
