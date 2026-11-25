# Masked model training lengths

Two of the four small masked diffusion models were trained on texts no longer than 512 tokens (DiffuGPT small) or 1,024 tokens (DiffuGPT medium, Masked Qwen3 0.6B). Masked Qwen2.5 Coder 0.5B's training length is not published. The two DiffuGPT models also cannot read past 1,024 tokens. The two Qwen models can address 32,768 and 40,960 positions, but nothing shows they were trained anywhere near that.

Question, 2026-11-25, after #238: how much text can each model take, and at what length was it trained? The position limits were already read from each checkpoint's config in "Masked model checkpoint candidates" (`research/masked-candidates`, #162). That report notes: "Configured maximum positions do not prove training coverage or useful extrapolation." This report fills that gap.

| Model | Most positions it can read | Length it was trained on | Short texts in training |
| --- | --- | --- | --- |
| DiffuGPT small | 1,024 (learned position table; a hard limit) | **512** | packed |
| DiffuGPT medium | 1,024 (hard limit) | **1,024** | packed |
| Masked Qwen2.5 Coder 0.5B | 32,768 (rotary positions) | **not published**; the training script's default is 512 | padded per batch, not packed (script defaults) |
| Masked Qwen3 0.6B | 40,960 (rotary positions) | **1,024** | padded per batch, not packed |

Our in-place prompts run 467 to 731 tokens. That is past DiffuGPT small's 512-token training length for every mask: its whole team sits in positions it never saw in training. The other three were trained at a length that covers these prompts, or (Masked Qwen2.5 Coder) may have been.

## DiffuGPT small and medium

- **Training length.** The paper's Table 4, "Training settings for different diffusion language models", lists DiffuGPT-S at 1000k steps, global batch size 256, context length **512**, and DiffuGPT-M at 160k steps, batch 1280, context length **1024**. Source: Gong et al., "Scaling Diffusion Language Models via Adaptation from Autoregressive Models", arXiv 2410.17891v3, Appendix B.2, Table 4 (<https://arxiv.org/html/2410.17891>).
- **Packing.** Section 4.1: "We use sequence packing, logits shifting, and 10K-step attention mask annealing to transform GPT2 to DiffuGPT." Same source.
- **Repository configs.** `LLaMA-Factory/examples/train_full/gpt2_full_ddm.yaml` (base `gpt2-medium`, `cutoff_len: 1024`, `packing: true`, `max_steps: 160000`) matches the paper's medium row. `gpt2_full_ddm-small.yaml` (base `gpt2`) also says `cutoff_len: 1024`, but with `max_steps: 10000` against the paper's 1000k, so it is an example config, not the recorded run. The paper's 512 stands; the repository's `gpt2_full_ddm-small-inf-512.yaml` inference config agrees with it. Source: <https://github.com/HKUNLP/DiffuLLaMA/tree/c17e897f6476c174b4623da594e4c65554f1613d/LLaMA-Factory/examples>.
- **Position limit.** `n_positions: 1024` and `n_ctx: 1024` in each checkpoint's `config.json` (diffugpt-s at `391d9850`, diffugpt-m at `ce62f897`), read from the local Hugging Face cache. GPT-2 learns one embedding per position, so position 1,025 does not exist.
- **Model cards.** Neither card states a training length. The medium card says "fine-tuned version of gpt2-medium on an unknown dataset", which the paper contradicts (FineWeb).

## Masked Qwen3 0.6B

- **Training length 1,024.** The dLLM repository's Tiny-A2D section: "To reproduce `Qwen3-0.6B-diffusion-mdlm-v0.1` (with MDLM & SFT), run the command below (about 10 hours on 64 A100s)", with `--script_path "examples/a2d/mdlm/sft.py"`, `--max_length 1024`, `--num_train_epochs 10` and output folder `epochs-10-bs-2048-len-1024`. The same section says "Here we show the exact commands we use to train". Source: `examples/a2d/README.md` at `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`, lines 243–256 (<https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/examples/a2d/README.md>).
- **Data and padding.** SFT on tulu-3-sft-mixture, smoltalk and the OpenCoder opc-sft stage 1 and 2 Python sets (same command). `sft.py` pads each batch (`DataCollatorForSeq2Seq(padding=True)`), sets `group_by_length=True`, and calls `post_process_dataset` to "truncate / filter long sequences if needed". The exact cut rule lives in `dllm.utils`, which was not read. Source: `examples/a2d/mdlm/sft.py` at the same commit.
- **Position limit.** `max_position_embeddings: 40960` in `config.json` at `c8d24a3f`. Positions are rotary, so a longer text runs, but quality past 1,024 tokens is untested.

## Masked Qwen2.5 Coder 0.5B

- **Training length not published.** The model card lists the opc-sft stage 1 and 2 datasets and links the W&B report; it has no training command. The repository's README lists this checkpoint only in an evaluation table, marked "(Reproduced)", with no training command. A code search of the repository finds the name in that README only.
- **The script default is 512.** `sft.py` sets `max_length: int = 512` in `DataArguments`. Unless the run overrode it, as the Qwen3 command does with 1,024, this checkpoint was trained at 512. That is an inference from the default, not a recorded value.
- **W&B report.** "dLLM Tiny-A2D" (<https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Tiny-A2D--VmlldzoxNTI2NTEzOA>) renders in the browser only; an automated fetch returned no text. It may record the run's `max_length`.
- **Position limit.** `max_position_embeddings: 32768` in `config.json` at `a284e895`.

## What this does not settle

- Masked Qwen2.5 Coder's training length. Reading the W&B run config in a browser would settle it.
- How `post_process_dataset` treats a training example longer than `max_length` (cut or dropped).
- Whether the uploaded Qwen3 checkpoint came from exactly the published command; the README calls it the reproduction command.
- How any model behaves on prompts longer than its training length. Nothing here was run.
