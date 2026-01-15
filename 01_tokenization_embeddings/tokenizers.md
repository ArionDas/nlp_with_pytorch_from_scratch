# Tokenization Methods

## What is tokenization?

Tokenization is the process of converting raw text into a sequence of discrete units called tokens. A **tokenizer** defines the rules and vocabulary used for this conversion, which directly affects vocabulary size, sequence length, and how rare or unseen text is handled. Tokenizers are essential for mapping text to ids that models can embed and process.

## What is a tokenizer?

A tokenizer is a deterministic (or probabilistic) function $f$ that maps text $T$ to a token sequence $[t_1, t_2, \dots, t_n]$. It includes:

- **Preprocessing** (normalization, Unicode handling, whitespace policy)
- **Segmentation** (how tokens are split or merged)
- **Vocabulary** (token-to-id mapping)

---

## Tokenization Methods (Technical Overview)

## 1) Whitespace Tokenization

**Idea:** Split text on whitespace boundaries.

**Algorithm:**
1. Normalize whitespace (optional).
2. Split on spaces/tabs/newlines.

**Notes:** Fast and simple but fails on punctuation, contractions, and languages without whitespace (e.g., Chinese, Japanese).

## 2) Word Tokenization (Rule-Based)

**Idea:** Use rules to split words, punctuation, and contractions.

**Algorithm:**
1. Normalize (case, Unicode forms).
2. Apply regex/rule set to separate punctuation (e.g., .,?!), clitics (e.g., "can't" → "ca" + "n't"), and numbers.
3. Emit tokens in order.

**Notes:** Language-dependent; rule quality strongly affects downstream performance.

## 3) Regex / Pattern Tokenization

**Idea:** Define token boundaries via regex patterns (e.g., words, numbers, emojis, URLs).

**Algorithm:**
1. Compile ordered regex patterns.
2. Scan input left-to-right.
3. Emit the longest matching pattern at each position.

**Notes:** Deterministic and controllable; brittle if patterns are incomplete.

## 4) Character Tokenization

**Idea:** Each Unicode code point is a token.

**Algorithm:**
1. Normalize Unicode (e.g., NFC).
2. Split into code points (not bytes).

**Notes:** Maximizes coverage and avoids OOV, but produces long sequences.

## 5) Byte-Level Tokenization

**Idea:** Treat input as bytes (0–255) and tokenize at the byte level.

**Algorithm:**
1. Encode text as UTF-8 bytes.
2. Each byte is a token (optionally merge via BPE/Unigram).

**Notes:** No OOV; robust to arbitrary text, but longer sequences unless merged.

## 6) WordPiece Tokenization

**Idea:** Greedy longest-match subword tokenization from a learned vocabulary.

**Paper (BERT):** "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"  
**Link:** https://arxiv.org/abs/1810.04805

**Training (high-level):**
1. Initialize vocabulary with characters and frequent words.
2. Iteratively add subwords that maximize likelihood under a language-model objective.

**Encoding:**
1. For each word, select the longest prefix in vocabulary.
2. Repeat on remaining suffix; mark continuations (e.g., "##ing").

**Notes:** Used by BERT; deterministic and efficient at inference.

## 7) Unigram Language Model Tokenization

**Idea:** Model tokenization as a probabilistic segmentation with a unigram LM.

**Paper (SentencePiece):** "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"  
**Link:** https://arxiv.org/abs/1808.06226

**Training (EM-style):**
1. Start with a large seed vocabulary.
2. Estimate token probabilities to maximize corpus likelihood.
3. Iteratively prune low-utility tokens.

**Encoding:**
1. Find the highest-likelihood segmentation (often via Viterbi/DP).

**Notes:** Used in SentencePiece (Unigram); supports multiple segmentations.

## 8) SentencePiece (BPE / Unigram)

**Idea:** Train subword tokenizers directly on raw text without pre-tokenization.

**Paper:** "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"  
**Link:** https://arxiv.org/abs/1808.06226

**Algorithm:**
1. Normalize text (optional).
2. Add a whitespace marker (e.g., ▁) to preserve boundaries.
3. Train BPE or Unigram on the marked stream.

**Notes:** Language-agnostic and robust to whitespace handling.

## 9) Morpheme-Based Tokenization (Linguistic)

**Idea:** Segment words into morphemes using linguistic rules or analyzers.

**Algorithm:**
1. Apply morphological analyzer / finite-state transducer.
2. Emit morphemes (roots, affixes).

**Notes:** Linguistically meaningful but requires language-specific tools.

## 10) Statistical / Hybrid Tokenization

**Idea:** Combine rule-based pre-tokenization with statistical subword models.

**Algorithm:**
1. Apply basic rules (punctuation, URLs, numbers).
2. Apply subword model (BPE/Unigram/WordPiece) on remaining spans.

**Notes:** Common in production to control edge cases and preserve domain tokens.

## 11) Byte-Pair Encoding (BPE)

**Idea:** Compression-inspired subword model that repeatedly merges the most frequent adjacent symbol pairs.

**Origin:**
**Paper:** "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)  
**Link:** https://arxiv.org/abs/1508.07909

**Training Algorithm:**
**Inputs:** training corpus $C$, target vocabulary size $V$, end-of-word marker (optional).

1. **Initialize symbols**
	- Split each word into a sequence of base symbols (characters or bytes).
	- Optionally append an end-of-word marker to preserve word boundaries.
	- The initial vocabulary is the set of all base symbols.

2. **Count adjacent pairs**
	- For all tokenized words in $C$, count frequencies of each adjacent symbol pair.

3. **Select best pair**
	- Find the pair $(x, y)$ with maximum frequency.
	- If no pair occurs more than once, stop early.

4. **Merge**
	- Replace all occurrences of $(x, y)$ with the merged symbol $xy$ in the corpus.
	- Add $xy$ to the vocabulary.

5. **Repeat** steps 2–4 until the vocabulary size reaches $V$ or no further merges are possible.

**Output:** ordered list of merge operations and the final vocabulary.

**Encoding Algorithm:**
**Inputs:** text $T$, learned merges list, base symbol set.

1. **Preprocess** $T$ into base symbols (characters or bytes), respecting the same normalization used during training.
2. **Apply merges** in the learned order, greedily combining symbol pairs that match each merge rule.
3. **Map symbols to ids** using the learned vocabulary; optionally add special tokens (BOS/EOS/UNK/PAD).

**Output:** sequence of subword token ids.

**Technical Notes:**

- **Determinism:** Given the same corpus and preprocessing, the merge list is deterministic.
- **Open vocabulary:** Unseen words can be decomposed into known subwords (or bytes).
- **Trade-off:** Larger $V$ yields longer subword units (shorter sequences), while smaller $V$ yields finer granularity.
- **Whitespace handling:** Tokenization is sensitive to how whitespace is represented (e.g., explicit space markers or byte-level encoding).

**Advantages:**

- Reduces out-of-vocabulary issues by composing rare words from frequent subwords.
- Compresses sequences compared to pure character-level tokenization.
- Learns data-driven segmentation without linguistic rules.

**Limitations:**

- Merge order can encode corpus-specific biases.
- Not linguistically motivated; subwords may split morphemes unintuitively.
- Sensitive to preprocessing choices (case, normalization, whitespace).

**Run Command:**

python 01_tokenization_embeddings/bpe_tokenizer_pytorch.py --text "This is a test run." --vocab_size 10000 --out_dir ./output
