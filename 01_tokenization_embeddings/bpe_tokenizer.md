# Byte-Pair Encoding (BPE) Tokenizer

## Overview

BPE is a data compression algorithm adapted for subword tokenization in NLP. It iteratively merges the most frequent character or character sequence pairs to build a vocabulary of subword units.

## Origin

**Paper:** "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)  
**Link:** https://arxiv.org/abs/1508.07909

Originally proposed for neural machine translation to handle rare and out-of-vocabulary words. Later adopted by GPT-2, RoBERTa, and other transformer models.

## Algorithm

1. **Initialize** vocabulary with individual characters
2. **Count** all adjacent symbol pairs in the corpus
3. **Merge** the most frequent pair into a new symbol
4. **Repeat** steps 2-3 until target vocabulary size reached

**Example:**
```
Corpus: "low low low lower lowest"
Initial: l o w  (chars + space marker)
Merge 1: lo + w → low
Merge 2: low + er → lower
Merge 3: low + est → lowest
```

## Key Properties

- **Subword segmentation:** Balances character-level and word-level tokenization
- **Open vocabulary:** Handles unseen words by decomposing into known subwords
- **Compression:** Reduces sequence length vs. character-level tokenization
- **Deterministic:** Same corpus always produces same merges

## Use Cases

**1. Machine Translation**
- Handles morphologically rich languages (German, Turkish, Finnish)
- Manages rare words and code-switching

**2. Large Language Models**
- GPT-2/GPT-3: Direct BPE on byte sequences
- RoBERTa: Modified BPE with byte-level encoding
- Reduces vocabulary size while maintaining coverage

**3. Multilingual Models**
- Shared subword vocabulary across languages
- Efficient handling of diverse scripts

**4. Domain-Specific NLP**
- Scientific text (chemical formulas, equations)
- Code generation (identifiers, operators)
- Social media (hashtags, mentions, emojis)

## Advantages

- No unknown tokens with byte-level BPE
- Learns data-driven segmentation
- Efficient for agglutinative languages
- Smaller vocabulary than word-level

## Limitations

- Sensitive to whitespace and formatting
- Merge order affects final vocabulary
- Not inherently linguistically motivated
- Longer sequences than word-level tokenization

## Implementation Notes

This implementation includes PyTorch integration for:
- Batch encoding with padding
- Tensor conversion for neural networks
- Special tokens (PAD, UNK, BOS, EOS)
- Compatible with `nn.Embedding` layers

## Usage

### Prerequisites
```bash
pip install torch
```

### Running the Tokenizer

**Execute the demo script:**
```bash
python 01_tokenization_embeddings/bpe_tokenizer_pytorch.py
```

**Programmatic usage:**
```python
from bpe_tokenizer_pytorch import BPETokenizerPyTorch

# Train tokenizer
corpus = "Your training text here..."
tokenizer = BPETokenizerPyTorch(vocab_size=256)
tokenizer.train(corpus)

# Single text encoding
text = "Machine learning is amazing!"
token_ids = tokenizer.encode(text, add_special_tokens=True)
decoded = tokenizer.decode(token_ids)

# Batch encoding with padding
texts = ["Hello world!", "Machine learning is great."]
batch = tokenizer.encode_batch(texts, max_length=20, add_special_tokens=True)
# Returns: {'input_ids': tensor, 'attention_mask': tensor}

# Save/load tokenizer
tokenizer.save("vocab.json")
tokenizer.load("vocab.json")

# Use with PyTorch embedding layer
from bpe_tokenizer_pytorch import TokenEmbedding
embed_layer = TokenEmbedding(tokenizer, embed_dim=128)
embeddings = embed_layer(batch['input_ids'])
```

### Output
The demo script demonstrates:
1. Training BPE on sample corpus
2. Single text tokenization and encoding
3. Batch encoding with attention masks
4. Token embedding layer integration
