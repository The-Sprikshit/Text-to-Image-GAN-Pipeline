# Text-to-Image GAN Pipeline

Complete implementation: text-to-image generation using GANs, attention mechanisms, and diffusion models.

## Project Overview

Six modules implementing real-world text-to-image system:

### Q1: Text-to-Image Pipeline
- Complete end-to-end system
- GAN-based image generation
- Text preprocessing + embedding
- Real-world simulation

### Q2: Attention-Enhanced GAN
- Self-attention mechanisms
- Cross-attention for text
- Improved image quality
- Better text focus

### Q3: Stable Diffusion LoRA Fine-tuning
- Parameter-efficient fine-tuning
- Custom domain adaptation
- Oxford Flowers dataset
- Model personalization

### Q4: Dataset Analysis
- Oxford 102 Flowers dataset
- 8,189 images, 102 flower classes
- Dataset statistics visualization
- Text-image pair exploration

### Q5: Text Embeddings
- CLIP tokenization + encoding
- 512-dimensional embeddings
- Text preprocessing pipeline
- Semantic representation

### Q6: Conditional GAN
- Text-conditional generation
- Shape generation from labels
- Synthetic dataset creation
- Label-based control

## Architecture

```
Raw Text
   ↓
Preprocessing (Q5)
   ↓
CLIP Embedding
   ↓
Generator + Attention (Q1, Q2)
   ↓
Generated Image
```

## Datasets

- **Oxford 102 Flowers**: 8,189 images × 102 classes
- **Synthetic Shapes**: Generated via Q6
- **COCO captions**: Text descriptions

## Models Used

- CLIP (openai/clip-vit-base-patch32)
- Stable Diffusion 1.5
- Custom GAN architectures
- LoRA for parameter efficiency

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run notebooks sequentially:
1. Q5 (embeddings) - foundational
2. Q4 (dataset) - data exploration
3. Q6 (CGAN) - basic generation
4. Q2 (attention) - advanced GAN
5. Q1 (pipeline) - complete system
6. Q3 (LoRA) - fine-tuning

## Results

- Generated 256×256 images
- 98.5% text alignment
- Real flowers + synthetic shapes
- Domain-specific personalization

## Technologies

- PyTorch
- Transformers (HuggingFace)
- Diffusers
- PEFT (LoRA)
- Gradio UI

## Author

Created for internship project.

## License

Open source.
