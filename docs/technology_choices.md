# Technology Choices

## Language & Framework

### Python 3.10+
- Standard for ML research
- Rich ecosystem (PyTorch, tokenizers, etc.)
- Matches existing research implementations

### Pure PyTorch
- **Why not Lightning?** More control, easier debugging, matches AGENTS.md style
- Explicit training loop visible in notebooks
- Easier to customize for diffusion-specific needs

### Marimo Notebooks
- **Why not Jupyter?** No hidden state, reactive updates, cleaner code
- Notebooks are Python files (version controllable)
- Can run headlessly with `marimo run`
- Built-in UI components (sliders, progress bars)

## Model Design

### GPT-2 Tokenizer
- Pre-trained, no need to train from scratch
- 50k vocab size - good balance of coverage and efficiency
- Widely used in research for comparison

### Non-Causal Transformer
- Bidirectional attention is essential for denoising
- Model sees entire sequence during training and inference
- Unlike autoregressive models, can infill anywhere

### Timestep Embedding
- Sinusoidal encoding + MLP (similar to DDPM)
- Encodes diffusion time as continuous signal
- Added to all token positions

## Diffusion Specifics

### Masked Diffusion (vs. Other Discrete Methods)
- Simpler than score-based discrete diffusion (SEDD)
- Better empirical results than uniform/absorbing state diffusion
- Cross-entropy loss is well-understood and stable

### Uniform Schedule (Default)
- No need to carefully tune schedule
- Works well for small models
- Can upgrade to cosine/anchored later

## Data

### TinyStories Dataset
- Simple vocabulary, short sequences
- Quick iteration during development
- Can upgrade to OpenWebText for larger experiments

### Tokenized-Only Storage
- Raw text ~2GB → Tokenized ~500MB
- Faster loading during training
- Trade-off: can't change tokenizer without re-processing

## Logging & Tracking

### Local JSON Logs
- Simple, portable, no dependencies
- Easy to load and visualize in notebooks
- No cloud account needed

### Why not WandB?
- Adds external dependency
- Overkill for single-developer project
- Can add later if needed

## Testing

### pytest
- Standard Python testing framework
- Fixtures for common test data
- Coverage reporting with pytest-cov

### Key Test Categories
- Schedule behavior at boundaries (t=0, t=max)
- Corruption preserves unmasked tokens
- Sampler never emits invalid token IDs
- Checkpoint save/load roundtrips correctly
