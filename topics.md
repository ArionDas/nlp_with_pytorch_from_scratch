# Complete NLP Implementation Curriculum for Research Scientists

## Tokenization & Embeddings
- [x] WordPiece Tokenization
- [x] Byte-Pair Encoding (BPE)
- [x] Unigram Language Model Tokenization
- [x] SentencePiece Tokenization
- [x] CBOW - Continuous Bag of Words
- [x] Skip-Gram 
- [x] "Token visualizer" to map words/chunks to IDs
- [x] One-hot vs learned embeddings: plot cosine distances

## Positional Embeddings
- [ ] Classic sinusoidal
- [ ] Learned positional embeddings
- [ ] RoPE
- [ ] ALiBi
- [ ] Animating a toy sequence being "position-encoded" in 3D
- [ ] Ablate positions—watch attention collapse

## Self-Attention & Multihead Attention
- [ ] hand-wire dot-product attention for one token
- [ ] scale to multi-head, plot per-head weight heatmaps
- [ ] mask out future tokens, verify causal property

## Transformers, QKV, & Stacking
- [ ] stack the Attention implementations with LayerNorm and residuals → single-block transformer
- [ ] generalize: n-block "mini-former" on toy data
- [ ] dissect Q, K, V: swap them, break them, see what explodes

## Sampling Parameters: temp/top-k/top-p
- [ ] code a sampler dashboard — interactively tune temp/k/p and sample outputs
- [ ] plot entropy vs output diversity as you sweep params
- [ ] nuke temp=0 (argmax): watch repetition

## KV Cache (Fast Inference)
- [ ] record & reuse KV states; measure speedup vs no-cache
- [ ] build a "cache hit/miss" visualizer for token streams
- [ ] profile cache memory cost for long vs short sequences

## Long-Context Tricks: Infini-Attention / Sliding Window
- [ ] implement sliding window attention; measure loss on long docs
- [ ] benchmark "memory-efficient" (recompute, flash) variants
- [ ] plot perplexity vs context length; find context collapse point

## Mixture of Experts (MoE)
- [ ] code a 2-expert router layer; route tokens dynamically
- [ ] plot expert utilization histograms over dataset
- [ ] simulate sparse/dense swaps; measure FLOP savings

## Grouped Query Attention
- [ ] convert your mini-former to grouped query layout
- [ ] measure speed vs vanilla multi-head on large batch
- [ ] ablate number of groups, plot latency

## Normalization & Activations
- [ ] hand-implement LayerNorm, RMSNorm, SwiGLU, GELU
- [ ] ablate each—what happens to train/test loss?
- [ ] plot activation distributions layerwise

## Pretraining Objectives
- [ ] train masked LM vs causal LM vs prefix LM on toy text
- [ ] plot loss curves; compare which learns "English" faster
- [ ] generate samples from each — note quirks

## Finetuning vs Instruction Tuning vs RLHF
- [ ] fine-tune on a small custom dataset
- [ ] instruction-tune by prepending tasks ("Summarize: ...")
- [ ] RLHF: hack a reward model, use PPO for 10 steps, plot reward

## Scaling Laws & Model Capacity
- [ ] train tiny, small, medium models — plot loss vs size
- [ ] benchmark wall-clock time, VRAM, throughput
- [ ] extrapolate scaling curve — how "dumb" can you go?

## Quantization
- [ ] code PTQ & QAT; export to GGUF/AWQ; plot accuracy drop

## Inference/Training Stacks
- [ ] port a model from HuggingFace to Deepspeed, vLLM, ExLlama
- [ ] profile throughput, VRAM, latency across all three

## Synthetic Data
- [ ] generate toy data, add noise, dedupe, create eval splits
- [ ] visualize model learning curves on real vs synthetic data

## Speculative Decoding
- [ ] implement draft model + verification loop for parallel token generation
- [ ] measure speedup vs autoregressive; plot acceptance rate curves
- [ ] ablate draft model size—find optimal quality/speed tradeoff

## Flash Attention & Sparse Attention
- [ ] code tiled attention with HBM-aware blocking; profile memory bandwidth
- [ ] implement sparse attention patterns (local, strided, global tokens)
- [ ] benchmark flash vs standard attention at varying sequence lengths

## Beam Search & Constrained Decoding
- [ ] hand-code beam search with length normalization, coverage penalties
- [ ] add grammar/regex constraints; force JSON/XML output structure
- [ ] visualize beam divergence and pruning decisions

## State Space Models (Mamba/SSM)
- [ ] implement selective state-space layer from scratch
- [ ] compare SSM vs attention on long sequences (10k+ tokens)
- [ ] plot memory usage and throughput vs transformer baseline

## Linear Attention Variants
- [ ] code Performer, Linformer, FNet-style approximations
- [ ] measure quality degradation vs quadratic attention
- [ ] profile speed gains on ultra-long contexts (100k+ tokens)

## Retrieval-Augmented Generation (RAG)
- [ ] build dense retriever (DPR-style) + vector database integration
- [ ] implement hybrid search (sparse BM25 + dense embeddings)
- [ ] ablate retrieval k, reranking, context truncation strategies

## Chunking & Indexing Strategies
- [ ] experiment with sentence, paragraph, sliding-window chunking
- [ ] build hierarchical indexing; measure recall@k across methods
- [ ] implement semantic caching for repeated queries

## Cross-Encoder Reranking
- [ ] train bi-encoder for fast retrieval + cross-encoder for reranking
- [ ] plot precision-recall curves; measure latency overhead
- [ ] compare to ColBERT-style late interaction

## Chain-of-Thought (CoT) & Reasoning
- [ ] implement zero-shot CoT, few-shot CoT, self-consistency sampling
- [ ] build scratchpad mechanism for multi-step math/logic
- [ ] parse reasoning chains; measure intermediate step correctness

## Tree-of-Thoughts & Planning
- [ ] code breadth-first thought exploration with pruning
- [ ] implement self-evaluation scoring for thought branches
- [ ] visualize search tree; compare to linear CoT

## ReAct & Tool Use
- [ ] build agent loop: reason → act → observe → iterate
- [ ] integrate mock tools (calculator, search, code executor)
- [ ] log tool call patterns; measure task success rate

## Curriculum Learning & Data Ordering
- [ ] implement difficulty-based curriculum (easy→hard)
- [ ] code domain mixing schedules; track per-domain loss
- [ ] ablate ordering strategies; plot convergence speed

## Continual Learning & Catastrophic Forgetting
- [ ] train on Task A, then Task B; measure Task A degradation
- [ ] implement EWC, replay buffers, parameter isolation
- [ ] plot forgetting curves across multiple task shifts

## Low-Rank Adaptation (LoRA) & PEFT
- [ ] code LoRA from scratch; inject into linear layers
- [ ] sweep rank r; plot accuracy vs trainable params
- [ ] compare LoRA, prefix-tuning, adapters on same task

## Distillation & Model Compression
- [ ] distill large teacher → small student with KL divergence
- [ ] implement progressive layer dropping, width pruning
- [ ] measure student performance vs compression ratio

## Multi-Task Learning
- [ ] build shared encoder with task-specific heads
- [ ] implement gradient balancing, task sampling strategies
- [ ] plot per-task learning curves; detect negative transfer

## Vision-Language Pretraining Basics
- [ ] implement CLIP-style contrastive loss (image-text pairs)
- [ ] build cross-modal attention (text attends to image patches)
- [ ] code image captioning decoder; evaluate on COCO subset

## Attention Visualization & Probing
- [ ] plot attention weight heatmaps; identify attention patterns
- [ ] build linear probes for intermediate layer representations
- [ ] ablate heads/layers; measure task impact

## Gradient-Based Attribution
- [ ] implement integrated gradients, saliency maps for tokens
- [ ] visualize which inputs drive specific outputs
- [ ] test adversarial robustness with gradient attacks

## Neuron Activation Analysis
- [ ] track which neurons fire for specific concepts
- [ ] implement causal tracing (edit activations, measure output change)
- [ ] build "neuron dictionary" by clustering activation patterns

## Constitutional AI & RLAIF
- [ ] generate critiques with LLM; use for synthetic preference data
- [ ] implement AI-as-judge for preference labeling
- [ ] compare RLAIF vs human RLHF on alignment tasks

## Direct Preference Optimization (DPO)
- [ ] code DPO loss (skip reward model training)
- [ ] compare DPO vs PPO: stability, sample efficiency, results
- [ ] ablate β parameter; plot reward-KL frontier

## Reward Model Training
- [ ] collect pairwise preferences; train Bradley-Terry reward model
- [ ] detect reward hacking; implement reward model ensembles
- [ ] plot reward model calibration curves

## Distributed Training
- [ ] implement data parallelism with DDP/FSDP
- [ ] code pipeline parallelism (split model across GPUs)
- [ ] profile communication overhead; optimize gradient all-reduce

## Mixed Precision & Gradient Accumulation
- [ ] enable FP16/BF16 training; handle loss scaling
- [ ] implement gradient accumulation for large effective batch
- [ ] measure throughput gains vs memory savings

## Checkpointing & Fault Tolerance
- [ ] implement activation checkpointing (gradient checkpointing)
- [ ] build resume-from-checkpoint with optimizer state
- [ ] simulate crash recovery; verify loss continuity

## Deduplication & Filtering
- [ ] implement MinHash LSH for near-duplicate detection
- [ ] code perplexity filtering, heuristic quality filters
- [ ] measure impact on downstream task performance

## Data Augmentation for NLP
- [ ] back-translation, synonym replacement, random masking
- [ ] generate paraphrases with model; create augmented dataset
- [ ] ablate augmentation ratio; plot generalization gains

## Bias Detection & Mitigation
- [ ] measure gender/race bias in embeddings (WEAT scores)
- [ ] implement debiasing techniques (projection, counterfactual)
- [ ] evaluate bias before/after mitigation

## Model Pruning
- [ ] implement magnitude pruning, movement pruning
- [ ] code structured pruning (prune entire heads/layers)
- [ ] plot sparse model accuracy vs FLOPs reduction

## Knowledge Distillation Variants
- [ ] progressive distillation (teacher → medium → student)
- [ ] feature-based distillation (match intermediate layers)
- [ ] on-policy distillation (student generates training data)

## ONNX Export & Optimization
- [ ] export PyTorch → ONNX → TensorRT/OpenVINO
- [ ] profile latency across frameworks
- [ ] measure accuracy drop from graph optimizations
