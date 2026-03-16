# Project Plan: Building a Small Diffusion Language Model (nanochat‑style)

This plan describes how to build a **nanochat‑style** repository that trains a *small diffusion language model* (DLM) using masked discrete diffusion.  It draws on recent research, provides implementation details, and discusses whether re‑writing the project in **Rust** instead of Python could be beneficial.  The plan is broken into phases to keep the scope manageable.  Each phase includes objectives, implementation steps, and optimization tips.  Citations refer to primary sources.

## 1. Background and Key Concepts

Diffusion language models replace the left‑to‑right next‑token prediction of autoregressive models with an iterative denoising process that starts from fully or partially masked sequences.  The reverse process learns to recover masked tokens given the noisy input, enabling **parallel token generation** and **bidirectional context**【491828077738106†L323-L340】【670718085216143†L45-L54】.  Recent research shows that **masked** diffusion (using a special `[MASK]` token as the absorbing state) performs best among discrete diffusion variants【844551102386334†L83-L112】 and that models like LLaDA, SEDD and MDLM achieve perplexity close to comparable autoregressive models【928863269607078†L39-L52】【670718085216143†L45-L54】.  

Important characteristics:

- **Forward process**: randomly mask tokens according to a schedule.  The fully corrupted state is a sequence of all `[MASK]` tokens【491828077738106†L370-L379】.  Token masking is the discrete analogue of adding noise【844551102386334†L83-L112】.
- **Reverse process**: a neural network (usually an encoder‑only Transformer) predicts the original tokens at masked positions.  Cross‑entropy loss on masked positions acts as a variational lower‑bound【928863269607078†L39-L52】【82830279971125†L67-L92】.
- **Sampling**: start with a masked sequence (prompt + masked continuation), iteratively unmask tokens based on model predictions and a sampling schedule.  Semi‑autoregressive (SAR) sampling or confidence‑based unmasking can accelerate inference【391669530464847†L67-L92】.

Diffusion LLMs promise faster parallel decoding, better controllability and the ability to edit or infill text anywhere【491828077738106†L414-L428】.  However, they often require more compute due to multiple denoising steps and still trail autoregressive models in some complex reasoning benchmarks【491828077738106†L431-L435】.

## 2. High‑Level Architecture

A nanochat‑style DLM repository should remain compact, transparent and hackable, similar to Karpathy’s **nanochat** but substituting the autoregressive objective with masked diffusion.  The core components:

1. **Tokenizer** – a Byte Pair Encoding (BPE) tokenizer (8k–16k tokens) or reuse GPT‑2’s 50k vocabulary for comparability.  Tokenization is needed for embedding and masking.
2. **Model** – a decoder‑only Transformer used as a denoising network.  Embedding layers map tokens to vectors; position embeddings include a time‑step or noise‑level embedding; transformer blocks process the masked sequence; an output head predicts token logits.  The model uses non‑causal self‑attention since it sees the entire sequence during denoising【491828077738106†L381-L386】.
3. **Noise schedule** – defines how masking probability `p(t)` evolves over diffusion steps.  Uniform scheduling (sample `p` ∼ U[0,1]) is simple for a small model; more advanced schedules (spindle, anchored, entropy‑bounded) can be added later【82830279971125†L100-L114】.
4. **Training loop** – sample time‑steps, mask tokens with probability `p`, feed the corrupted sequence and time embedding into the model, compute cross‑entropy loss on masked positions, and update model parameters.  Training is essentially masked language modeling but with varied mask ratios【928863269607078†L39-L52】.
5. **Sampling loop** – given a prompt, append a masked continuation of fixed length, then iteratively unmask tokens.  Use either random token positions per step or a heuristic (confidence‑based unmasking).  After each step, freeze confident tokens and optionally remask low‑confidence ones【391669530464847†L67-L90】.  Provide options for SAR sampling (unmask small blocks sequentially) or full diffusion sampling.

## 3. Repository Structure

A suggested directory layout balances clarity and minimalism:

```
nanochat_diffusion/
│
├── README.md            # project overview and instructions
├── pyproject.toml       # Python package specification
│
├── tokenizer/
│   ├── train_tokenizer.py
│   └── tokenizer.json   # trained BPE vocabulary
│
├── model/
│   ├── transformer.py   # transformer and embeddings
│   ├── diffusion_head.py# output head for masked denoising
│   └── schedule.py      # time/noise schedule helpers
│
├── diffusion/
│   ├── forward.py       # masking functions
│   ├── sampling.py      # sampler implementing SAR and full diffusion
│   ├── loss.py          # masked cross‑entropy loss
│   └── utils.py         # logging, parameter schedules
│
├── data/
│   ├── prepare_text.py  # dataset download and formatting
│   └── datasets/        # small corpora (TinyStories, OpenWebText subset)
│
├── train/
│   ├── train_base.py    # training script for base model
│   └── train_sft.py     # script for instruction tuning (phase 2)
│
├── eval/
│   └── eval_ppl.py      # perplexity and sample evaluation
│
└── app/
    ├── chat.py          # CLI chat interface
    └── web.py           # minimal web interface (optional)
```

Each file should have less than ~200 lines to keep it understandable; docstrings and comments explain the algorithm.

## 4. Phases of Implementation

### Phase 1 – Minimal Base Model

**Objective**: implement a working masked diffusion LM on a small corpus (e.g., TinyStories or a small OpenWebText subset).  

**Steps**:

1. **Prepare environment**
   - Use Python ≥3.10, PyTorch with GPU support, and install dependencies via `pip` or `conda` (Lightning can help manage loops).  For reproducibility, provide a `requirements.txt`/`pyproject.toml`.
   - Train a BPE tokenizer on the chosen corpus (8k–16k vocab).  You may reuse GPT‑2’s vocabulary to compare perplexity.【287009971483210†L339-L378】 emphasises that Python’s ecosystem (NumPy, PyTorch, etc.) and large community make rapid prototyping easier than low‑level languages.

2. **Implement model components**
   - **Embedding layers**: token embedding, positional embedding, and *time‑step embedding*.  The time embedding encodes the mask ratio; you can use sinusoidal or learned embeddings and add them to token embeddings.
   - **Transformer architecture**: start with 6–12 layers, hidden dimension 512–768, 8–12 heads.  Use non‑causal self‑attention (no mask) so that the model uses bidirectional context【491828077738106†L381-L386】.  Add layer normalization and dropout.
   - **Output head**: a linear layer mapping hidden states to vocabulary logits.

3. **Implement diffusion process**
   - **Masking schedule**: for each sample, draw a random time `t∈[0,1]` or sample discrete step `k` up to `T` (e.g., 1000 steps).  Convert `t` to mask probability `p` via a schedule; simple linear or cosine schedules work.  Mask tokens in the input with probability `p`.
   - **Loss function**: compute cross‑entropy only on positions where tokens were masked.  This training objective is equivalent to a Rao‑Blackwellized variational lower‑bound【928863269607078†L39-L52】.

4. **Training loop**
   - Use a data loader to stream tokenized sequences.  For each batch: sample `p`, mask tokens, embed with time; feed through the model; compute loss and backpropagate.  Use gradient accumulation to fit into GPU memory if necessary.  
   - Monitor training with `wandb` or simple printouts.  Evaluate perplexity on a held‑out validation set.
   - Provide a simple script to launch training via the command line or a Slurm script (see MDLM’s `scripts/train_owt_mdlm.sh`【273218031018821†L417-L427】).

5. **Sampling and demonstration**
   - Implement iterative sampling: initialize a `[MASK]` sequence with desired length; at each step, feed into the model; choose tokens for unmasking and fill them with predicted argmax (or sample with temperature); update confidence scores; remask low‑confidence positions if using SAR sampling【391669530464847†L67-L92】.
   - Provide a CLI to supply prompts and view intermediate denoising steps.  Visualizing the unmasking process helps users understand diffusion.

**Optimization tips**:

- Start with a **tiny model** (e.g., 25 M parameters) and short sequence length (e.g., 256–512 tokens) to ensure fast iteration.  Scale up once the pipeline works.
- Use **mixed precision** training (`torch.cuda.amp`) to speed up training.
- Leverage the `ddpm_cache` sampler (caching intermediate computations) from MDLM to reduce generation time by 3–4×【273218031018821†L340-L353】.
- Gradient accumulation and careful batch sizing help when training on limited hardware【273218031018821†L424-L427】.

### Phase 2 – Instruction Tuning and Chat Interface

**Objective**: fine‑tune the base DLM to follow instructions and engage in chat.

1. **Prepare an instruction dataset**: use publicly available instruction‑following datasets (e.g., Alpaca, Dolly) or create a synthetic dataset.  Convert each instruction–response pair into a sequence where the response region is masked; the model learns to generate the response conditioned on the instruction【491828077738106†L394-L405】.

2. **Training modifications**: only mask tokens in the response during fine‑tuning, leaving the instruction untouched.  The loss remains cross‑entropy on masked positions.

3. **Sampling modifications**: given a prompt, append a fully masked continuation of a specified length and run the diffusion sampler.  Provide options to adjust generation length and number of denoising steps.  Optionally incorporate semi‑AR sampling for faster decoding.

4. **Chat interface**: implement a simple CLI (using Python’s `readline`) where the user enters instructions and receives model outputs.  Add a web UI (e.g., Streamlit or a minimal Flask app) later.  Logging conversation histories helps debug output quality.

### Phase 3 – Advanced Features and Optimizations

Once the base system works, gradually add more sophisticated features:

1. **Better masking schedules** – implement anchored or entropy‑bounded schedules that prioritize masking high‑entropy tokens later【82830279971125†L100-L114】.
2. **Confidence‑based unmasking** – instead of random positions, unmask tokens whose predicted confidence exceeds a threshold.  This heuristic improves efficiency【391669530464847†L67-L92】.
3. **Semi‑autoregressive decoding** – unmask small blocks sequentially (SAR), leading to 25–30× faster decoding than full diffusion【273218031018821†L384-L393】.
4. **Hybrid AR/Diffusion models** – explore mixing autoregressive and diffusion objectives as in HART or Eso‑LMs【670718085216143†L107-L112】.
5. **Longer sequences** – incorporate local attention or hierarchical transformers to scale sequence length (e.g., 2048 or 4096 tokens).  Inference can maintain speed via block diffusion sampling.
6. **RL‑trained unmasking policies** – integrate reinforcement learning to learn the unmasking policy rather than hand‑crafted heuristics【391669530464847†L67-L92】.

## 5. Datasets and Evaluation

Start with **TinyStories** for quick iteration; progress to **OpenWebText** or **FineWeb** subsets to improve generalization.  Preprocessing scripts should perform sentence splitting, tokenization and sequence length adjustment.  For evaluation:

- **Perplexity** on held‑out text (use the `eval_ppl.py` script).  MDLM shows diffusion models can achieve perplexity close to autoregressive baselines【928863269607078†L39-L52】.
- **Generation quality** via human inspection or automatic metrics (BLEU, ROUGE).  Provide sample outputs at different time steps to illustrate denoising.
- **Inference speed**: compare wall‑clock time and tokens per second for diffusion vs. semi‑AR sampling【273218031018821†L340-L353】.

## 6. Should You Rewrite in Rust?

Rust promises **speed**, **memory safety**, and **concurrency**.  Its compiler enforces rules that prevent data races and eliminates the overhead of a garbage collector【287009971483210†L423-L456】.  For AI workloads requiring low‑latency inference or integration with systems code, Rust can offer high performance.  However, **Python** remains the dominant choice for machine learning because of its **simplicity**, **readability**, and **rich ecosystem of libraries**—PyTorch, NumPy, scikit‑learn—which are highly optimized under the hood【287009971483210†L339-L378】.  Most academic codebases, including MDLM and SEDD, are written in Python; this makes reproducing research easier【287009971483210†L382-L390】.

If you consider rewriting components in Rust:

- **Library support**: Rust’s machine‑learning ecosystem is growing (e.g., `tch-rs` bindings for PyTorch, `ndarray`, `linfa`)【287009971483210†L465-L470】, but it lacks the maturity of Python’s frameworks.  Training deep networks still requires GPU libraries that are primarily built for Python/C++.
- **Interoperability**: You could use Rust to implement performance‑critical parts (data loading, tokenization, noise sampling) via `PyO3` or `tch-rs`, exposing them to Python.  This hybrid approach leverages Rust’s speed while retaining Python’s flexibility【287009971483210†L423-L456】.
- **Development velocity**: Prototyping diffusion architectures involves experimentation with hyperparameters and model tweaks; Python’s ease of use and community examples accelerate iteration.  Rust’s stricter type system and compile‑time checks slow down rapid changes.

Given these trade‑offs, **Python** is the recommended language for your nanochat‑diffusion prototype.  After confirming correctness, you may optimize specific components in Rust or C++ if profiling shows Python becomes a bottleneck.  For example, implementing the masking forward process and sampling loops in Rust and binding them back to Python could reduce overhead, while leaving model training in PyTorch.

## 7. Additional Learning Resources

| Resource | Type | Notes |
| --- | --- | --- |
| **Simple and Effective Masked Diffusion Language Models (MDLM)** – project page & code | Paper + code | Provides the original MDLM paper and implementation; includes scripts for training and sampling【928863269607078†L39-L52】【273218031018821†L290-L335】. |
| **EmergentMind: Masked Diffusion Language Models** | Overview article | Explains diffusion LM principles, forward/reverse processes, masking schedules and sampling strategies【82830279971125†L47-L63】【82830279971125†L100-L114】. |
| **DigitalOcean: What are Text Diffusion Models?** | Tutorial | Offers an accessible introduction, contrasts diffusion with autoregressive models, describes LLaDA training and sampling details【491828077738106†L323-L340】【491828077738106†L370-L379】 and notes advantages/limitations【491828077738106†L414-L428】【491828077738106†L431-L435】. |
| **Janu Verma’s blog – Diffusion Models III: Language Models** | Detailed blog | Walks through discrete diffusion from first principles, compares noise strategies, and emphasises masking as the best discrete corruption; includes PyTorch code examples【844551102386334†L83-L112】【844551102386334†L146-L149】. |
| **Oxen.ai – How to Train Diffusion for Text from Scratch** | Article + code | Goes through training diffusion models for text with Score Entropy Discrete Diffusion (SEDD); includes dataset preparation and code modifications【712410917976851†L20-L49】. |
| **Hugging Face blog – Diffusion Language Models: The New Paradigm** | Overview | Summarises diffusion LM research, including LLaDA and hybrid models; highlights throughput and commercial achievements【670718085216143†L45-L54】【670718085216143†L98-L104】. |
| **Awesome‑DLMs GitHub repository** | Collection | Curated list of diffusion LM papers, codebases, and tutorials. |
| **Diffusion Language Models Explained (with live coding)** | Video | A YouTube session that explains diffusion LMs and demonstrates live coding (referenced via Reddit post). |
| **Large Language Diffusion with mAsking (LLaDA) paper** | Paper | Introduces LLaDA, an 8B‑parameter diffusion LLM; includes training details such as mask‑sampling and step scheduling【491828077738106†L370-L379】【670718085216143†L98-L104】. |

## 8. Conclusion

A nanochat‑style small diffusion language model is feasible and educational.  By replacing autoregressive next‑token prediction with masked diffusion, you can explore modern generative modelling techniques while maintaining a compact, understandable codebase.  Start with a simple Python implementation using a lightweight Transformer and masked cross‑entropy loss; iteratively expand functionality with improved sampling strategies, instruction tuning, and hybrid architectures.  Although Rust offers compelling performance features, the established ML ecosystem, rapid prototyping advantages and abundant research code make Python the pragmatic choice for initial development.  You can later optimize bottlenecks in Rust once you have a working baseline.
