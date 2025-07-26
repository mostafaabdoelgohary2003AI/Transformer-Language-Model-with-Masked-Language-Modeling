# Transformer Language Model with Masked Language Modeling

A PyTorch implementation of a transformer-based language model using masked language modeling (MLM) training, similar to BERT. Trained on "The Hitchhiker's Guide to the Galaxy" text dataset.

## Overview

This project implements a transformer architecture for language modeling using the masked language modeling approach. The model learns to predict masked tokens in text sequences, enabling it to understand contextual relationships and generate coherent text representations.

## Features

- **Transformer Architecture**: Multi-head attention and feed-forward networks
- **Masked Language Modeling**: BERT-style self-supervised learning
- **Custom Text Processing**: Handles arbitrary text datasets
- **Token Masking Strategy**: Configurable masking ratios and patterns
- **GPU Acceleration**: CUDA support for efficient training
- **Flexible Tokenization**: Compatible with Hugging Face tokenizers
- **Batch Processing**: Efficient training with variable sequence lengths

## Architecture

### Model Components
```python
TransformerLM(
├── Embedding Layer (vocab_size → embed_dim)
├── Transformer Encoder
│   ├── Multi-Head Attention (num_heads)
│   ├── Feed-Forward Network
│   └── Layer Normalization
├── Output Projection (embed_dim → vocab_size)
└── Cross-Entropy Loss
)
```

### Key Parameters
- **Vocabulary Size**: Determined by tokenizer (typically ~30K tokens)
- **Embedding Dimension**: 512 (configurable)
- **Number of Heads**: 8 (configurable)
- **Number of Layers**: 6-12 (configurable)
- **Sequence Length**: 128 tokens (configurable)
- **Masking Ratio**: 20% of tokens (configurable)

## Dataset

- **Source**: "The Hitchhiker's Guide to the Galaxy" by Douglas Adams
- **Format**: Plain text file processed into paragraphs
- **Preprocessing**: Tokenization, padding, and masking
- **Training Strategy**: Self-supervised learning with masked tokens

## Requirements

```python
torch>=1.7.0
transformers>=4.0.0
numpy>=1.19.0
random
```

## Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install torch transformers numpy
   ```

2. **Download the dataset**:
   - Place "Hitchhiker Guide to the Galaxy.txt" in the specified path
   - Or update the `dataset_path` variable with your text file

3. **Configure paths**:
   ```python
   dataset_path = "/path/to/your/text/file.txt"
   ```

## Usage

### Running the Notebook

1. **Open** `transformer-llm-masking.ipynb` in Jupyter or Google Colab
2. **Execute cells sequentially** to:
   - Load and preprocess the text data
   - Initialize the transformer model
   - Train with masked language modeling
   - Evaluate model performance

### Key Functions

```python
# Load and preprocess dataset
paragraphs = load_dataset(dataset_path)

# Tokenize and apply masking
masked_ids, target_tokens, mask_indices = tokenize_and_mask(
    paragraph, mask_token_id, tokenizer
)

# Initialize transformer model
model = TransformerLM(vocab_size, embed_dim, num_heads, num_layers, mask_token_id)

# Compute masked language modeling loss
loss = compute_loss(logits, target_tokens)
```

### Training Process

1. **Text Preprocessing**: Load and clean text data
2. **Tokenization**: Convert text to token IDs
3. **Masking**: Randomly mask 20% of tokens
4. **Forward Pass**: Predict masked tokens
5. **Loss Computation**: Cross-entropy on masked positions
6. **Backpropagation**: Update model parameters

## Model Architecture Details

### Transformer Encoder
- **Multi-Head Attention**: Captures long-range dependencies
- **Position Encoding**: Implicit through learned embeddings
- **Feed-Forward Networks**: Non-linear transformations
- **Residual Connections**: Gradient flow and training stability
- **Layer Normalization**: Training stability and convergence

### Masking Strategy
```python
# Dynamic masking with configurable ratio
num_to_mask = max(1, int(0.2 * num_tokens))
mask_indices = random.sample(range(num_tokens), num_to_mask)
```

## Training Configuration

### Hyperparameters
- **Learning Rate**: 1e-4 (Adam optimizer recommended)
- **Batch Size**: 16-32 (adjust based on GPU memory)
- **Sequence Length**: 128 tokens
- **Masking Ratio**: 20% of tokens
- **Warmup Steps**: 1000-2000 steps
- **Training Steps**: 10K-100K steps

### Optimization
- **Optimizer**: Adam with weight decay
- **Learning Rate Schedule**: Warmup + cosine decay
- **Gradient Clipping**: Prevent gradient explosion
- **Mixed Precision**: FP16 for memory efficiency

## Applications

### Text Understanding
- **Text Classification**: Fine-tune for sentiment analysis
- **Named Entity Recognition**: Extract entities from text
- **Question Answering**: Answer questions based on context
- **Text Summarization**: Generate concise summaries

### Text Generation
- **Creative Writing**: Generate story continuations
- **Code Completion**: Predict next code tokens
- **Language Translation**: Cross-lingual understanding
- **Dialogue Systems**: Conversational AI applications

## Evaluation Metrics

### Training Metrics
- **Masked Language Modeling Loss**: Primary training objective
- **Perplexity**: Model's uncertainty on predictions
- **Accuracy**: Percentage of correctly predicted masked tokens

### Downstream Tasks
- **GLUE Benchmark**: General language understanding
- **Reading Comprehension**: SQuAD-style datasets
- **Text Classification**: Domain-specific tasks

## Advanced Features

### Custom Tokenization
```python
# Add special tokens for domain-specific vocabulary
tokenizer.add_special_tokens({
    "additional_special_tokens": ["[CUSTOM]", "[DOMAIN]"]
})
```

### Dynamic Masking
```python
# Different masking strategies
def advanced_masking(tokens, mask_ratio=0.15):
    # 80% mask, 10% random, 10% unchanged
    pass
```

## Performance Optimization

### Memory Efficiency
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Mixed Precision Training**: Reduce memory usage
- **Dynamic Padding**: Variable sequence lengths in batches

### Computational Efficiency
- **Model Parallelism**: Distribute across multiple GPUs
- **Gradient Checkpointing**: Trade compute for memory
- **Efficient Attention**: Linear attention mechanisms

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch size or sequence length
- **Slow convergence**: Check learning rate and masking ratio
- **Poor predictions**: Increase model size or training data
- **Tokenizer errors**: Verify vocabulary and special tokens

### Performance Tips
- Monitor training loss curves for convergence
- Use learning rate scheduling for better results
- Implement early stopping to prevent overfitting
- Save model checkpoints regularly

## Fine-tuning Guide

### Domain Adaptation
1. **Load Pre-trained Model**: Start with general language model
2. **Domain Data**: Prepare domain-specific text corpus
3. **Continued Pre-training**: MLM on domain data
4. **Task Fine-tuning**: Supervised learning for specific tasks

### Best Practices
- Start with smaller learning rates for fine-tuning
- Use task-specific data augmentation
- Monitor validation metrics closely
- Apply regularization techniques (dropout, weight decay)

## Future Enhancements

- [ ] Implement attention visualization tools
- [ ] Add support for longer sequences (up to 512 tokens)
- [ ] Experiment with different masking strategies
- [ ] Implement ELECTRA-style replaced token detection
- [ ] Add multi-lingual support
- [ ] Create interactive text generation interface
- [ ] Implement knowledge distillation for model compression

## Research Applications

### Academic Use Cases
- **Language Model Analysis**: Understanding attention patterns
- **Transfer Learning**: Pre-training for downstream tasks
- **Computational Linguistics**: Studying language representations
- **NLP Research**: Baseline for new architectures

### Industry Applications
- **Search Engines**: Better text understanding
- **Chatbots**: Improved conversational AI
- **Content Generation**: Automated writing assistance
- **Code Intelligence**: Programming language understanding

## Contributing

We welcome contributions in the following areas:
- Model architecture improvements
- Training efficiency optimizations
- Evaluation framework enhancements
- Documentation and tutorials
- Bug fixes and code quality improvements

## References

- **Attention Is All You Need**: Vaswani et al., 2017
- **BERT**: Devlin et al., 2018
- **RoBERTa**: Liu et al., 2019
- **Transformers Library**: Hugging Face documentation
- **PyTorch Tutorials**: Official PyTorch transformer guide

## License

This project is available under the MIT License. Educational and research use encouraged.

---

*Exploring the power of self-supervised learning and transformer architectures for natural language understanding.* 
