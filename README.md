# NLP with PyTorch from Scratch

From-scratch implementations of core NLP and LLM techniques in PyTorch, covering tokenization, embeddings, and recent research papers.

## Repository Structure

```
.
├── 01_tokenization_embeddings/
│   ├── corpus/english_corpus.txt
│   ├── bpe_tokenizer_pytorch.py
│   ├── wordpiece_tokenizer_pytorch.py
│   ├── unigram_tokenizer_pytorch.py
│   ├── sentencepiece_tokenizer_pytorch.py
│   ├── word2vec_pytorch.py
│   ├── embedding_comparison.py
│   ├── token_visualizer.py
│   ├── utils.py
│   └── tokenizers.md
├── paper_implementations/
│   ├── SDPO/
│   │   ├── config.py
│   │   ├── policy_model.py
│   │   ├── reward_model.py
│   │   ├── reprompting.py
│   │   ├── self_distillation.py
│   │   ├── trainer.py
│   │   └── example.py
│   └── reason_in_13_params/
│       ├── config.py
│       ├── tiny_lora.py
│       ├── lora_xs.py
│       ├── model.py
│       ├── grpo.py
│       └── example.py
├── requirements.txt
├── topics.md
└── LICENSE
```

## 1. Tokenization and Embeddings

### 1.1 Tokenizers

Four subword tokenization algorithms, each implemented as a standalone class with `train`, `tokenize`, `encode`, and `decode` methods. All support special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`) and JSON vocabulary serialization.

**BPE Tokenizer** (`bpe_tokenizer_pytorch.py`) -- `BPETokenizerPyTorch`
- Iteratively merges the most frequent adjacent token pair until target vocab size is reached.
- Tokenization applies learned merges in order. Used by GPT-2/GPT-3.

**WordPiece Tokenizer** (`wordpiece_tokenizer_pytorch.py`) -- `WordPieceTokenizerPyTorch`
- Merges are selected by likelihood score: `freq(ab) / (freq(a) * freq(b))`, favoring pairs of rare tokens that co-occur.
- Inference uses greedy longest-match within the learned vocabulary. Includes `[CLS]`, `[SEP]`, `[MASK]` tokens. Used by BERT.

**Unigram Tokenizer** (`unigram_tokenizer_pytorch.py`) -- `UnigramTokenizerPyTorch`
- Top-down approach: starts with a large initial vocabulary, then iteratively prunes tokens that minimize corpus likelihood loss.
- Tokenization uses Viterbi dynamic programming to find the highest-probability segmentation.

**SentencePiece Tokenizer** (`sentencepiece_tokenizer_pytorch.py`) -- `SentencePieceTokenizerPyTorch`
- Language-agnostic; operates on raw Unicode input with no pre-tokenization step.
- Represents whitespace using the `\u2581` marker. Supports both BPE and Unigram backends. Used by T5, LLaMA.

### 1.2 Embeddings

**Word2Vec** (`word2vec_pytorch.py`)
- Implements both CBOW (`CBOWModel`) and Skip-gram (`SkipGramModel`) as `nn.Module` subclasses.
- CBOW predicts a center word from averaged context embeddings; Skip-gram predicts context words from the center word.
- Includes `Word2VecDataset` for generating training pairs from a configurable context window.

**Embedding Comparison** (`embedding_comparison.py`)
- Compares one-hot, random dense, and learned (Word2Vec-trained) embeddings side by side.
- Provides cosine similarity matrices and k-nearest-neighbor queries via `compute_embedding_similarity` and `find_nearest_embeddings`.

### 1.3 Utilities

**Token Visualizer** (`token_visualizer.py`)
- Terminal-based visualization of tokenization output with ANSI-colored tokens and frequency tables.
- Includes `CharTokenizer` and `WordTokenizer` for baseline comparison.

**Shared Utilities** (`utils.py`)
- Text processing: `split_into_words`, `get_word_frequencies`, `extract_unique_chars`.
- BPE helpers: `count_pair_frequencies`, `merge_pair_in_words`, `apply_merges_to_word`.
- WordPiece helpers: `get_wordpiece_word_splits`, `compute_wordpiece_pair_scores`, `merge_wordpiece_pair`.
- Unigram helpers: `build_initial_unigram_vocab`, `viterbi_tokenize`, `compute_unigram_loss`, `prune_unigram_vocab`.
- SentencePiece helpers: `normalize_for_sentencepiece`, `get_sentencepiece_word_freqs`, `denormalize_from_sentencepiece`.
- Vocabulary I/O: `build_initial_vocab`, `invert_vocab`, `save_tokenizer_data`, `load_tokenizer_data`.
- Batching and embeddings: `batch_encode_with_padding`, `TokenEmbeddingLayer(nn.Module)`, `compute_embedding_similarity`, `find_nearest_embeddings`.

### 1.4 Training Corpus

`corpus/english_corpus.txt` -- approximately 50 sentences covering diverse topics (governance, animals, geography, time, daily life) for vocabulary building and frequency estimation.

---

## 2. Paper Implementations

### 2.1 SDPO -- Self-Distilled Policy Optimization

Implementation of "Reinforcement Learning via Self-Distillation" (arXiv:2601.20802).

**Core loop:**
1. Generate multiple rollouts per prompt.
2. Score with a verifiable reward model.
3. Select high-reward trajectories as demonstrations.
4. Reprompt the model with demonstrations to form a teacher distribution.
5. Update the policy with a combined loss: policy gradient + self-distillation KL + KL penalty.

**Modules:**

| File | Class / Purpose |
|------|----------------|
| `config.py` | `SDPOConfig`, `SelfDistillationConfig`, `RepromptingConfig` -- all hyperparameters as dataclasses. |
| `policy_model.py` | `PolicyModel` -- wraps a HuggingFace causal LM with a student and an EMA teacher copy. Provides `generate_rollouts` and `update_teacher_ema` (exponential moving average: phi = (1-beta)*phi + beta*theta). |
| `reward_model.py` | `RewardModel` (abstract base), `VerifiableRewardModel` (regex pattern matching, binary reward), `RuleBasedCodeRewardModel` (composable code quality checks), `MathRewardModel` (LaTeX `\boxed{}` extraction and verification). |
| `reprompting.py` | `Reprompter` -- constructs demonstration-augmented prompts from successful rollouts using configurable templates, with optional `<think>` tag stripping and truncation. |
| `self_distillation.py` | `SelfDistillationLoss` -- supports forward KL, reverse KL, and JSD (controlled by alpha). Optional top-k distillation with tail bucket. `PolicyGradientLoss` -- clipped ratio objective (PPO-style) with importance sampling. |
| `trainer.py` | `SDPOTrainer` -- orchestrates the full training loop: rollout generation, reward computation, reprompting, loss computation, gradient update, and EMA teacher update. Tracks per-step metrics (total loss, PG loss, distillation loss, KL penalty, mean reward). |
| `example.py` | End-to-end usage demo. |

### 2.2 Learning to Reason in 13 Parameters (TinyLoRA)

Implementation based on arXiv:2602.04118 (Morris et al., Meta FAIR + CMU). Demonstrates that RL (GRPO) enables learning with extreme parameter budgets -- 13 trainable parameters achieve 91% on GSM8K with Qwen2.5-7B.

**Key idea:** decompose the weight update through the frozen SVD of each linear layer, then parameterize the residual with a single trainable scalar passed through a unique frozen random projection per module.

**Modules:**

| File | Class / Purpose |
|------|----------------|
| `config.py` | `TinyLoRAConfig` (frozen_rank, trainable_dim, n_tie, target_modules), `LoRAXSConfig`, `GRPOConfig`. |
| `tiny_lora.py` | `TinyLoRALinear(nn.Module)` -- replaces a frozen linear layer. Computes truncated SVD of the original weight (U, Sigma, V are frozen buffers), generates frozen random projections P per module, and holds a trainable vector v (typically dim 1). Forward: `output = Wx + alpha * U Sigma R V^T x` where `R = sum_i v_i P_i`. `TinyLoRAParameterGroup` -- manages weight tying across modules: groups of `n_tie` modules share the same v while maintaining distinct P matrices. |
| `lora_xs.py` | `LoRAXSLinear(nn.Module)` -- LoRA-XS baseline. Same SVD decomposition but the trainable parameter is a full r-by-r matrix R instead of a scalar-projected sum. Parameter count: O(r^2) per module. |
| `model.py` | `apply_tiny_lora(model, config)` -- freezes the base model, finds target linear layers, injects `TinyLoRALinear` wrappers with shared parameter groups. `apply_lora_xs(model, config)` -- same for LoRA-XS. `count_trainable_params(model)` -- returns parameter count and byte sizes (fp32, bf16). `freeze_base_model`, `print_adapter_summary`. |
| `grpo.py` | `GRPOBatch` dataclass (prompt_ids, completion_ids, rewards, attention_mask). `GRPOTrainer` -- Group Relative Policy Optimization: generates k completions per prompt, computes group-normalized advantages `A = (r - mean) / std`, applies clipped policy gradient loss, optional KL penalty against a reference model. No value function / critic required. |
| `example.py` | Full pipeline demo with a toy transformer: apply TinyLoRA, count params, train with GRPO. |

**Parameter count reference (Qwen2.5-7B, 28 layers, 7 target modules per layer = 196 modules):**

| Method | Config | Trainable Params |
|--------|--------|-----------------|
| TinyLoRA | u=1, n_tie=15 | 13 |
| TinyLoRA | u=1, n_tie=1 | 196 |
| LoRA-XS | rank=1 | 196 |
| LoRA | rank=1 | ~3M |
| Full fine-tuning | -- | 7.6B |

---

## Requirements

```
matplotlib
pandas
numpy
```

PyTorch and HuggingFace Transformers are additionally required for paper implementations.

## License

MIT -- Copyright (c) 2026 Arion Das
