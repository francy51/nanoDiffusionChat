# Development Roadmap

## Phase 1: Core Implementation (Current)

- [x] Project scaffolding
- [x] Configuration system
- [x] Tokenizer wrapper
- [x] Diffusion schedule and corruption
- [x] Model architecture (embeddings, transformer, denoiser)
- [x] Training loop with checkpointing
- [x] Basic sampler
- [x] Notebook workflow

### Remaining for Phase 1
- [ ] Fix any bugs from initial testing
- [ ] Complete test coverage
- [ ] Train first baseline model
- [ ] Document generation quality

## Phase 2: Improvements

### Sampling Enhancements
- [ ] Confidence-based unmasking
- [ ] Semi-autoregressive (SAR) sampling
- [ ] Temperature annealing schedules

### Schedule Improvements
- [ ] Anchored schedules
- [ ] Entropy-bounded masking
- [ ] Learnable schedules

### Model Improvements
- [ ] Flash attention for efficiency
- [ ] Mixture-of-experts layers
- [ ] Longer sequence support (local attention)

## Phase 3: Instruction Tuning

- [ ] Prepare instruction dataset
- [ ] Fine-tune on instruction-response pairs
- [ ] Chat interface notebook
- [ ] CLI chat script

## Phase 4: Scaling

- [ ] OpenWebText training
- [ ] Larger model configs
- [ ] Distributed training support
- [ ] Model evaluation benchmarks

## Future Considerations

### Hybrid AR/Diffusion
- Combine autoregressive and diffusion objectives
- Potentially best of both worlds

### RL-based Unmasking
- Learn unmasking policy with reinforcement learning
- Adaptive step selection

### Rust Optimization
- Critical path profiling
- Consider Rust for:
  - Data loading
  - Tokenization
  - Sampling loops
