# zeos-torch — a PyTorch machine backend for ZEOS

A demo in the ZEOS repository and a member of its uv workspace. `TorchMachine` implements
the `MachineBackend` protocol over a Hugging Face causal LM, running one forward pass per
decode step on PyTorch, with one KV cache per job.

```bash
uv sync --all-packages                 # from the repository root
cd demo/torch-backend
uv run zeos-torch fetch-model          # ~1GB, Qwen/Qwen2.5-0.5B-Instruct into models/
uv run --package zeos-torch pytest     # the contract suite, M0 and this backend
```

`ZEOS_TORCH_MODEL` points the tests at weights elsewhere. Without weights the backend's
tests skip and the scripted backend's still run.

## The model

Any Hugging Face causal LM whose every layer is full softmax attention over a cache that
can be cut back to an arbitrary position. The loader refuses anything else
(`DenseAttentionRequired`): sliding-window and linear-attention layers have neither an
attention row over the whole context nor a truncatable cache. It loads with
`attn_implementation="eager"` in float32 on CPU by default.

## Units

The kernel counts words; the model counts ids. As in coop-count's llama machine, each
kernel word records how many ids back it (`spans`) and how many of those are chat framing
folded in around it (`framing`). Each model position also records the kernel offset it
belongs to (`owner`), and framing ids belong to the word they are folded into. A position
therefore sits in exactly one kernel word, and so in exactly one block.

## What is real and what is not

| | Status |
| --- | --- |
| Decoding | **Real.** A forward pass per step, greedy, deterministic on CPU. |
| `trunc`, `splice` | **Real.** The KV is cut back and the dropped positions recomputed on the next decode. `splice` reports the downstream words it invalidates. |
| `fork` | **Real, physical.** The child gets a deep copy of the parent's KV, not a shared one. |
| Control tokens | **Real.** Block padding and the tokenizer's special tokens are made unselectable at the sampler unless the kernel enables control tokens. Text injected into a context is tokenized with special tokens split, so injected text cannot spell chat framing. |
| End of generation | An end-of-generation id is an `EXIT` request, never an exception. |
| The mask | **Refused.** `set_mask` raises `MaskViolation` for a mask that narrows the live blocks, as coop-count's llama machine does, rather than accept a mask it does not enforce. |
| Attention | **Synthetic.** `attention` is `None` and the step carries a hint: `descriptor` until the job has spoken, `self` after. |
| Grammar | **None.** Nothing constrains the model to the syscall ABI, so most commands a small model produces parse as `MALFORMED`. |
