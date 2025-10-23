# Encoder-Decoder Architecture for Sequence-to-Sequence Translation

A PyTorch implementation of an LSTM-based encoder-decoder architecture for neural machine translation tasks, specifically designed for English-to-Spanish translation.

## Overview

This project implements a classic sequence-to-sequence (Seq2Seq) model using LSTM networks for translating English sentences to Spanish. The architecture consists of an encoder that processes the input sequence and a decoder that generates the translated output sequence.

## Features

- **LSTM-based Architecture**: Uses multi-layer LSTM networks for both encoding and decoding
- **BERT Tokenization**: Leverages BERT's pre-trained tokenizer for robust text processing
- **Special Token Handling**: Implements SOS (Start of Sequence) and EOS (End of Sequence) tokens
- **Gradient Clipping**: Includes gradient clipping to prevent exploding gradients
- **Training Pipeline**: Complete training loop with loss computation and accuracy tracking

## Architecture Components

### 1. TEXT_TO_EMBEDD Class
Handles text preprocessing and tokenization:
- Uses BERT's uncased tokenizer
- Adds special tokens `[SOS]` and `[EOS]`
- Converts text to token IDs with padding and truncation
- Returns input/output IDs and vocabulary size

### 2. Encoder
Multi-layer LSTM encoder that:
- Processes embedded input sequences
- Returns output sequences and hidden/cell states
- Supports configurable number of layers, hidden dimensions, and dropout

### 3. Decoder
Multi-layer LSTM decoder that:
- Takes encoder's hidden states as initial states
- Processes target sequences with teacher forcing
- Generates predictions for each time step

### 4. Encoder_Decoder_Architecture
Main model combining encoder and decoder:
- Embedding layer for token representation
- Linear output layer projecting to vocabulary size
- Softmax activation for probability distribution

## Requirements

```bash
torch
transformers
tensorflow  # Only for keras layers (can be removed if not needed)
```

## Installation

```bash
pip install torch transformers
```

## Usage

### Training the Model

```python
# Define your parallel corpus
english_sentences = [
    "The early bird catches the worm and starts its day.",
    "A gentle rain fell all night long, nourishing the newly planted garden.",
    # ... more sentences
]

spanish_sentences = [
    "Al que madruga, Dios le ayuda, y comienza su día.",
    "Una suave lluvia cayó toda la noche, nutriendo el jardín recién plantado.",
    # ... more translations
]

# Train the model
model, tokenizer = train(
    epochs=1000,
    sentences=english_sentences,
    translated_sentences=spanish_sentences,
    lr=0.001
)
```

### Model Parameters

The default configuration includes:
- **Embedding Dimension**: 128
- **Encoder Layers**: 3
- **Encoder Hidden Size**: 128
- **Encoder Dropout**: 0.3
- **Decoder Layers**: 3
- **Decoder Hidden Size**: 128
- **Decoder Dropout**: 0.2
- **Learning Rate**: 0.001

## Model Architecture

```
Input Sentence → Tokenization → Embedding Layer → Encoder LSTM
                                                       ↓
                                                  Hidden States
                                                       ↓
Target Sentence → Tokenization → Embedding Layer → Decoder LSTM → Linear Layer → Predictions
```

## Training Details

- **Loss Function**: CrossEntropyLoss (ignoring padding tokens)
- **Optimizer**: Adam
- **Gradient Clipping**: Maximum norm of 1.0
- **Teacher Forcing**: Uses ground truth target sequences during training
- **Metrics**: Tracks both loss and token-level accuracy

## Dataset

The implementation includes 30 parallel English-Spanish sentence pairs covering various everyday scenarios. The sentences range from simple phrases to complex descriptions.

## Output

During training, the model prints:
```
Epoch: 0, loss: 10.08, accuracy: 0.028
Epoch: 100, loss: , accuracy: X.XXXX
...
```

## Limitations

- Fixed maximum sequence length (determined by tokenizer)
- Teacher forcing during training (no inference mode implemented)
- Limited to the provided vocabulary
- No beam search or advanced decoding strategies

## Future Improvements

- Add inference/translation function without teacher forcing
- Implement beam search decoding
- Add attention mechanism
- Support for variable-length sequences without excessive padding
- Model checkpointing and saving
- Validation set evaluation
- BLEU score computation
