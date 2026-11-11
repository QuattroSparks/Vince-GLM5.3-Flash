# Credits: Vince GLM5.3 Flash

**Thank you.** This recipe is a small layer on a very large amount of other people's work, and it exists only because
they chose to share theirs. Every token it serves passes through code, weights and ideas that someone else built,
measured and gave away. This page is here to say whose, precisely, and to celebrate them.

## First of all: Mia and Ash

- **[Mia's AI Lab](https://github.com/MiaAI-Lab)** wrote the recipe this is a fork of,
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (Apache-2.0). What is forked is her **Python engine** setup for four Sparks (`ENGINE=python`: TensorFold v0.6.0 with
  her 106 patches). A git detail, so nobody is misled by a name: she keeps that setup on a branch called `zig-engine`; that branch also carries her newer Zig engine, and there is no separate Python branch. The fork starts at
  her commit `697d8e9` there, and runs and changes the Python engine only; the Zig engine is never started. The
  checkpoint, the 106 patches, the four-Spark switchless ring, the scripts, the documentation and sparkDash, the
  dashboard the published figures were measured with, are hers. She made four Sparks on a cable loop serve this model
  at all; this fork only made one part of it faster. Thank you, Mia, for building it in the open and for the care in
  every line of your own credits.
- **Ash Hart** ([ashhart](https://github.com/ashhart)) **and the TensorFold contributors** built
  [TensorFold](https://github.com/ashhart/TensorFold): the engine, the GLM-5.3-Flash CUDA kernels, DFlash2 drafting
  with exact verification, the server. Every file this fork changes is a TensorFold file. Thank you, Ash, for an
  engine that is fast, exact and readable enough for others to build on.

Mia's own [`CREDITS.md`](../CREDITS.md) and [`NOTICE`](../NOTICE) are kept unchanged and stay the reference for the
recipe underneath; this file adds what this fork took, from whom, and where it went.

If something here is missing or credited wrongly, please open an issue: it will be corrected, gladly.

## Whose work this is

| Idea or code | Whose | Where it is used here | Source | Licence |
|---|---|---|---|---|
| The recipe: the scripts, the 106 TensorFold patches, the four-Spark switchless ring start, the documentation | **Mia's AI Lab** | Everything outside `vince/`, `start-vince.sh` and `stop-vince.sh`, unchanged. `start-vince.sh` ends by running her `start-tp4-switchless.sh` | [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) @ `697d8e9` | Apache-2.0 |
| TensorFold v0.6.0: the engine, the GLM-5.3-Flash CUDA engine and its kernels, DFlash2 drafting with exact verification, the server | **Ash Hart** ([ashhart](https://github.com/ashhart)) **and the TensorFold contributors** | Every file in `vince/overlay` is a TensorFold file (as Mia's patches leave it) with this fork's changes. `TF_GLM_PF_FP8` and `TF_GLM_PF_KQ` call TensorFold's FP8 prompt kernels `quantize_rows` and `prefill_matmul8` as they are; `pf_xchg.py` reuses the HCA, GID and route helpers of `tensorfold/cuda/roce.py` | [ashhart/TensorFold](https://github.com/ashhart/TensorFold) | Apache-2.0 from v0.6.0; MIT notice for code written before it (`NOTICE`) |
| The checkpoint (EXL3, routed experts at 4 bits a weight) | **Mia's AI Lab** | The weights every mode serves | [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) | Apache-2.0 |
| The row split of a prompt chunk between the ranks, each rank's DSA front on its own rows, the two micro-batches a chunk (her patches 0098-0106; the code is hers, the own-row front and micro-batch designs are **drowzeys'**) | **Mia's AI Lab**; design: **drowzeys** | `hcsplit.py`, `own_front.py`, `microbatch.py` in `vince/overlay` are her files with this fork's changes: the exchange is routed through them | Mia's `CREDITS.md`; [drowzeys/TensorFold](https://github.com/drowzeys/TensorFold/tree/glm53-tp4-2026-10-05) | Apache-2.0 (Mia's code; drowzeys' branch is Apache-2.0 too) |
| The options `TF_GLM_PREFILL_ROWS`, `TF_GLM_PROMPT_PARTIALS=bf16` (her patch 0106), `TF_GLM_MSA=1` (her patch 0101, with design points from **drowzeys** and **BertholomusAI (Albert Lee)**) and `TF_GLM_PROMPT_MICROBATCH_MIN_ROWS` (her patch 0104) | **Mia's AI Lab** | Settings in `vince/modes.sh`; this fork only chose their values and measured them | Mia's `CREDITS.md`, `README-Mia.md` | Apache-2.0 |
| The RoCE all-gather relayed round a four-Spark ring, and the queue pairs' ACK timeout (her patches 0099, 0100) | **Mia's AI Lab** | In `roce_proxy.c`, the file `pf_xchg_proxy.c` is a copy of; decode still runs on hers, unchanged | Mia's `NOTICE` | Apache-2.0 |
| RoCEnante: the one-shot RDMA proxy and its GPU kernel protocol (stages, flags, memory orderings) | **b12x** by **local-inference-lab** (Luke Alonso and the b12x contributors) | `pf_xchg_proxy.c` is a modified copy of `b12x/comm/roce/_roce_proxy.c` @ `8a99d639` as TensorFold and Mia's AI Lab had already modified it; `pf_xchg.cu` follows the protocol as TensorFold's `roce.cu` implements it. The file's header and `NOTICE` list the changes | [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) | Apache-2.0 |
| The idea of keeping the RoCE proxy thread on the GB10's big cores | **Christopher Owen** | `TF_GLM_DEC_CPUS=big` (`dec_tune.py`): applied here as an affinity mask on a rank's whole process; no code of his | [christopherowen/dgx-spark-networking](https://github.com/christopherowen/dgx-spark-networking) | Apache-2.0 |
| The proxy thread confined to the cores whose `cpu_capacity` lies above the midpoint (`resolve_proxy_cpus` / `place_proxy_thread`), and a `dmb oshld` barrier after observing a NIC-written flag | **Christopher Owen** | `pf_xchg_proxy.c`: `TF_GLM_PF_XCHG_PROXY_CPU=big` and `TF_GLM_PF_XCHG_DMB=1`, rewritten in a shorter form after his `sparknet/oneshot/_roce_proxy.c`. Both are off by default and not measured | same | Apache-2.0 |
| The "ring4" host relay: on a four-box cable loop the neighbours' proxies relay to the box across | **Christopher Owen** | One of the two designs the exchange's relay follows (idea only) | same | Apache-2.0 |
| Direct stage: skip the stage copy when the input already is the send slot (branch `feat/sparknet-pdl-direct-stage`) | **Christopher Owen** | Tried as `TF_GLM_PF_SI` with `TF_GLM_PF_XCHG_PART`: slower on this exchange, so it is in the image, off. The code is written for this exchange | same | Apache-2.0 |
| Switchless-ring collectives: one-shot RDMA writes through a CPU proxy, the opposite rank's halves relayed by the neighbours both ways; and the per-link measurement method (`ib_write_bw` per directed link, both PCIe domains: `micro/ring_bw.sh`) | **Andy Shu** ([blaesus](https://github.com/blaesus)), dsgx10 (its repository is named dsgb10) | The design of the exchange's relay and how the cables were measured. Ideas only, no code | [blaesus/dsgb10](https://github.com/blaesus/dsgb10) (`ds41f-4xgx10-switchless/docs/ring.md`, `docs/findings.md`) | MIT |
| Multi-user batching and wide steps on the same kind of ring | **Pico (Patrick Moorhead)**, pico-dgx-ds4, a fork of dsgx10 | Read for design ideas (`CHANGES.md`, `ds41f-4xgx10-switchless/docs/wide64.md`). Nothing was taken into code | [Picom-code/pico-dgx-ds4](https://github.com/Picom-code/pico-dgx-ds4) | MIT |
| Patches inherited through Mia's recipe: the spill tier (0088) and the Zig engine's `0014` patches by **Robert Wojciechowski** ([wojo](https://github.com/wojo), [@rwojo](https://x.com/rwojo)); the shared-prefix copy (0071), the display reservation (0072), compact before evict (0074) and the Zig `0015` / `0016` patches by **ezoushen** | **Robert Wojciechowski**, **ezoushen** | In Mia's `patches/` and `zig/patches/`, unchanged here | Mia's `CREDITS.md` | Apache-2.0 |
| Decode loads of the routed experts and the L2 prefetch (Mia's patches 0047, 0046), adapted from glm53-tensorfold-spark | **Jay Leaton** | `exl3.cu`, `exl3.cpp` and `exl3_mm.py` in `vince/overlay` still carry patch 0047's load path, untouched | [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark); Mia's `NOTICE` | Apache-2.0 |
| The chunked form of the KDA recurrence (Mia's patches 0012, 0014, 0039, after flash-linear-attention) | **Mia's AI Lab**; algorithm: the **flash-linear-attention** authors | `kda_chunk.cu`, `kda_chunk.cpp`, `kda_chunked.py` in `vince/overlay` | Mia's `CREDITS.md`; [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | Apache-2.0; MIT |
| The DFlash2 drafter (weights), its architecture, and TensorFold's port of it | **IncoAI**; **Z Lab** (z-lab/dflash); **Ash Hart and the TensorFold contributors** | Drafts every decode round. `TF_GLM_PF_TAP_TAIL` changes only how often a prompt chunk updates it | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2); [z-lab/dflash](https://github.com/z-lab/dflash) | The drafter: its model card states `apache-2.0` (read 2026-11-12), while Mia's `NOTICE` records CC BY-NC-ND 4.0 from an earlier reading: the card governs, read it before any commercial use. z-lab/dflash: MIT |
| GLM-5.3-Flash: the model | **Z.ai** ([zai-org](https://huggingface.co/zai-org)) | The model being served | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | MIT (its model card governs) |
| The EXL3 format and its converter, exllamav3 | **turboderp** | The format of the routed experts' weights | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | MIT |
| tool-eval-bench, the 69-scenario tool-call benchmark sparkDash runs | **SeraphimSerapis** | Measured v2.5's optional tool-call rules (`TF_GLM_TOOL_RULES`); nothing of it is copied here. The rules' text and the switch are this recipe's own | [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) | MIT |
| sparkDash's DecodeBench protocol and its prose prompt | **Mia's AI Lab** | `vince/tools/dbench.py` and `rep.py` reproduce the protocol; the prose prompt is quoted, the code prompts are read from a sparkDash checkout at run time | [MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash) | Apache-2.0 |
| Read for ideas, nothing taken | **kindlingai** (gr8); **Mia's AI Lab** (GLM-5.3-Flash-DGX-Sparks-1104, `docs/speed/RESULTS.md`: fixed prompts, rounds and ms a round as the decode gate); **othexmr** (RiNGSiDE) | The way decode was measured | [kindlingai/gr8](https://github.com/kindlingai/gr8), [MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104), [othexmr/ringside](https://github.com/othexmr/ringside) | MIT; Apache-2.0; MIT |
| Everyone else Mia's recipe builds on: drowzeys, BertholomusAI (Albert Lee), BadAd84, Matt Mastracci (Kindling AI), ThomasWadeZ, johnwhited, desy0305, abhicnv007, Alexbob0, kky42, meleesciony, jdecker76, m-naoki-m, eduffd, lukaszraczylo, kerpopule, jschmied and the others, and the runtime stack (NVIDIA's PyTorch container, PyTorch, Triton, NCCL, rdma-core, ...) | as named in Mia's [`CREDITS.md`](../CREDITS.md) | In the image and the patches underneath, unchanged | Mia's `CREDITS.md`, `NOTICE` | as listed there |

## What is new here

This is the whole of this fork's own work. Each item is a change to TensorFold files that Mia's patches had already
shaped, so none of it stands without the rows above.

- **The RoCE ring exchange** (`TF_GLM_PF_XCHG`; `pf_xchg_proxy.c`, `pf_xchg.cu`, `pf_xchg.cpp`, `pf_xchg.py`, and its
  use in `hcsplit.py`): an all-to-all exchange mode for the four-Spark ring, added to a copy of the RoCEnante proxy.
  Every rank writes its block straight into each neighbour's memory, the block for the rank across goes half through
  each neighbour, and the DSA layers' own-row shares get a second exchange on their own stream. It replaces NCCL for
  the row split's prompt exchanges. Replies stay bit-identical.
- **FP8 activations for the dense prompt projections** (`TF_GLM_PF_FP8`): TensorFold's own FP8 prompt kernels wired
  into the GLM engine's 4-bit dense prompt projections. It changes the rounding of prompt arithmetic.
- **bf16 expert outputs** (`TF_GLM_PF_EY16`): the routed experts' prompt outputs written as bf16 by a `Y16` variant of
  the EXL3 down-projection kernel (`exl3.cu`), read by `glue.py`'s combine. It changes the rounding of prompt
  arithmetic.
- **The same-bits prefill switches of v2 and v2.2**: `TF_GLM_PF_TAP_TAIL`, `TF_GLM_PF_LEAN`, `TF_GLM_PF_PART16`,
  `TF_GLM_PF_OWN_ROUTER`, `TF_GLM_PF_ZC` (v2), `TF_GLM_PF_KQ`, `TF_GLM_PF_XCHG_BCAST`, `TF_GLM_PF_FILL_WHOLE` (v2.2).
  They remove passes over memory and work every rank repeated; replies are identical with and without them.
- **Two decode switches** (v2.1): `TF_GLM_DEC_MSG` (a decode round's message to the other ranks as one RoCE frame)
  and `TF_GLM_DEC_CPUS` (a rank's threads on the big cores; the idea is Christopher Owen's, above).
- **The short-prompt switches of v2.3**: `TF_GLM_PF_TAIL_MERGE` (a prompt's last 1-63 rows run in the chunk before
  them) and `TF_GLM_PF_BALANCE` (a prompt longer than one chunk runs in equal chunks), changes to `multi.py`,
  `forward.py`, `decode.py`, `hcsplit.py` and `microbatch.py`. No outside code. (v2.3's third setting,
  `TF_GLM_PROMPT_MICROBATCH_MIN_ROWS=960`, is a value of Mia's option.)
- The measurements, checks and tools in `vince/tools` and `vince/results`, and `start-vince.sh`.

The engineering (code, measurements, documentation) was done with [Claude Code](https://claude.com/claude-code)
(Anthropic) under Vince's direction.

## v2.4 (2026-11-11)

- 32-row verify windows: the width is **TensorFold's** (ashhart/TensorFold #2257, v1.2.6); the code that serves it
  here (`TF_GLM_MAX_ROWS`, `TF_GLM_WIDE_GRAPHS`, copy drafts) is **Mia's AI Lab's** and **TensorFold's**, already in
  the image. This recipe only raises the settings and `start.sh`'s check.
- Malformed tool calls: the fault and its rule (such a call stays text) are **TensorFold's** #2801 / #2831, reported
  there by **@devopsfortunato**. Their fix is Zig; the Python here (`server/tools.py`, `cuda/reply_text.py`) is
  written for this engine, no code copied.
- Heat bands: **TensorFold's** `TF_HEAT_HIGH` / `TF_HEAT_LOW` (Nemotron on CUDA and the GB10 runbook), designed from
  **@Bizuayeu's** GLM-5.3-Flash measurements on three Sparks (TensorFold #339, #988). `heat.py` is written for this
  engine's multi-stream server, no code copied.
- `TF_GLM_EXL3_DEC_ORDER` (patch 0090) is **Lukasz Raczylo's** ([lukaszraczylo](https://github.com/lukaszraczylo)); measured here on 2026-11-11, no gain at TP=4, left off.
- The four-Spark ring notes read for this version are **@Bizuayeu's** (TensorFold #1971, #2352, #2569).

## v2.5 (2026-11-11)

- The optional tool-call rules (`TF_GLM_TOOL_RULES`): the rules' text and the switch are this recipe's own, off by
  default. They were measured with **SeraphimSerapis'**
  [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) (MIT), as **Mia's AI Lab's** sparkDash runs
  it; nothing of the benchmark is copied here. Its 69 scenarios are what showed which habits were the model's.
- The quality comparison with stock uses sparkDash's quality bench (**Mia's AI Lab**), unchanged.
- The server already honoured `parallel_tool_calls: false`: that is **TensorFold's** and **Mia's AI Lab's** work.

## Optional speech-to-text (v2.4)

- `vince/extras/parakeet/`: **NVIDIA's** Parakeet TDT 0.6B v3 (CC BY 4.0) served through **Hugging Face
  Transformers**; the small server and installer are this recipe's own. The remedy for the slow load (clone the
  memory-mapped weights before moving them to the GPU) was found on this cluster.

## With gratitude

To everyone named on this page and in Mia's credits: thank you. A few words each, because a table does not say it.

- **Mia**: for the recipe, the checkpoint, sparkDash, and for answering the question "can four Sparks do this?" with
  working code.
- **Ash Hart and the TensorFold contributors**: for the engine underneath everything, and for the pace at which it
  keeps getting better (v2.4 here is three of your ideas).
- **drowzeys** and **Albert Lee (BertholomusAI)**: for the prompt-side designs Mia's patches carry; the fast prefill
  stands on them.
- **Luke Alonso and the b12x contributors**: for RoCEnante. The ring exchange is a mode added to your proxy.
- **Christopher Owen**: for the big-core idea, the barrier, the ring relay and the direct stage, all written up so
  clearly that they could be tried in an afternoon.
- **Andy Shu**: for showing how a switchless ring should exchange data, and how to measure every cable honestly.
- **Pico (Patrick Moorhead)**: for sharing the multi-user work on the same kind of ring; reading it helped.
- **Robert Wojciechowski**, **ezoushen**, **Jay Leaton**, **Lukasz Raczylo**, **BadAd84** and every contributor to
  Mia's recipe: your patches run here untouched, every day.
- **@Bizuayeu** and **@devopsfortunato**: for the measurements and the bug reports upstream that became v2.4's heat
  bands and tool-call fix.
- **SeraphimSerapis**: for a tool-call benchmark sharp enough to tell the model's habits from the build's.
- **Z.ai**, **IncoAI**, **Z Lab**, **turboderp**, the **flash-linear-attention** authors, **kindlingai**,
  **othexmr**, **NVIDIA** and **Hugging Face**: for the model, the drafter, the format, the algorithm, the reading
  and the runtime that all of this needs before it can start.

If this recipe saves you time, the thanks belong to them first.

## Licences

- This fork's own changes: Apache-2.0, like Mia's recipe ([`LICENSE`](../LICENSE)). [`NOTICE`](../NOTICE) keeps Mia's
  notices and adds a section stating what this fork changed (Apache-2.0, section 4).
- The files in `vince/overlay` are modified TensorFold v0.6.0 files as Mia's 106 patches leave them, so they stay
  under TensorFold's licences (Apache-2.0, and its MIT notice for code written before v0.6.0), with Mia's and her
  contributors' changes under Apache-2.0. `vince/glm53-vince-v2.5.patch` shows every change this fork made to them.
- b12x, christopherowen/dgx-spark-networking: Apache-2.0 (code adapted, as listed above). dsgx10, pico-dgx-ds4: MIT
  (ideas only, no code).
- Every licence in the table was read again at its source on 2026-11-12 (the repositories' licence files and the
  model cards on Hugging Face).
- The model files are downloaded from Hugging Face by Mia's `scripts/prepare.sh` and are not part of this repository;
  each has its own licence (table above, `README-Mia.md` "License").
