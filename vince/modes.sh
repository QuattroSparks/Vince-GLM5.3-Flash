# Vince GLM5.3 Flash: the three builds (README.md "Modes"). scripts/local.sh sources this file after this setup's own
# values (vince/local.sh.example); Mia's scripts/config.sh sources scripts/local.sh first, and the environment wins
# over both: `VINCE_MODE=exact ./start-vince.sh restart`, `TF_GLM_MSA=0 ./start-vince.sh restart`.
#
#   fast  (default) the RoCE ring exchange + four prompt-precision switches (replies not bit-identical to Mia's stock
#                   build; quality-checked) + v2's five same-bits switches + v2.2's 12,288-row chunks and three more
#                   same-bits switches + v2.3's three short-prompt settings + v2.1's two decode switches + v2.4's
#                   32-row verify windows and heat bands
#   exact           the RoCE ring exchange + 6,144-row prompt chunks + v2.1's two decode switches + v2.4's heat bands:
#                   every greedy reply
#                   bit-identical to Mia's stock build
#   stock           Mia's image and settings, unchanged
VINCE_IMAGE=${VINCE_IMAGE:-tensorfold-glm53:vince-v2.4}   # Mia's tensorfold-glm53:v0.6.0 + vince/overlay (vince/build-image.sh)
ENGINE=python                         # Mia's TensorFold v0.6.0 Python engine: the engine vince/overlay modifies
case "${VINCE_MODE:-fast}" in
  fast|exact)
    IMAGE=$VINCE_IMAGE
    PREPARE=0                         # scripts/prepare.sh would build Mia's stock image under this name: never run it
    export TF_GLM_PREFILL_ROWS=6144   # prompt chunks of 6,144 rows (exact mode; fast mode takes 12,288 below)
    export TF_GLM_PF_XCHG=1           # the row split's prompt exchanges over the RoCE ring exchange instead of NCCL
    # v2.1: two decode switches that change no arithmetic (fast and exact alike)
    export TF_GLM_DEC_CPUS=big        # every thread of a rank on the ten big cores (the round loop, the RoCE proxy)
    export TF_GLM_DEC_MSG=1           # a round's message to the other ranks as one RoCE frame, not two NCCL gathers
    # v2.4: heat bands, TensorFold's TF_HEAT_HIGH / TF_HEAT_LOW for this engine; no reply changes (fast and exact alike)
    export TF_GLM_HEAT_HIGH="${TF_GLM_HEAT_HIGH:-78}"   # the head holds a prompt's next chunk while its GPU reads above
    export TF_GLM_HEAT_LOW="${TF_GLM_HEAT_LOW:-72}"     #   78 C, until 72 C or less; decode is never held; 0 / 0: off
    if [[ "${VINCE_MODE:-fast}" == fast ]]; then
      # v1: four prompt-precision switches (they change the rounding of prompt arithmetic)
      export TF_GLM_PF_FP8=1          # FP8 activations for the 4-bit dense prompt projections
      export TF_GLM_PROMPT_PARTIALS=bf16   # Mia's switch: the row split's partial sums as bf16
      export TF_GLM_MSA=1             # Mia's switch: the msa.cu prompt sparse attention
      export TF_GLM_PF_EY16=1         # the routed experts' prompt outputs as bf16
      # v2: five switches that keep every bit of the fast build (replies identical to v1's, checked)
      export TF_GLM_PF_TAP_TAIL=1     # a prompt chunk updates the drafter only with its last ring of rows
      export TF_GLM_PF_LEAN=1         # two passes nothing read
      export TF_GLM_PF_PART16=1       # the bf16 partials written by the matmul / combine itself (no cast pass)
      export TF_GLM_PF_OWN_ROUTER=1   # the MoE router on a rank's own rows, picks shipped with the rows
      export TF_GLM_PF_ZC=1           # hc_post reads the peers' partials in the exchange's receive slots (no copy)
      # v2.2: larger prompt chunks and three more switches, replies still identical to v1's fast mode
      export TF_GLM_PREFILL_ROWS=12288     # the routed experts read their weights once per 6,144-row half (a 12,288-row
                                           # start hung once, on 2026-11-08, and not since; if a prompt ever hangs,
                                           # 8192 is the next best: -2% at 32k)
      export TF_GLM_PF_KQ=1           # before a KDA layer each rank quantizes only its own rows and ships the FP8 bytes
      export TF_GLM_PF_XCHG_BCAST=1   # a row broadcast staged once, not once a peer
      export TF_GLM_PF_FILL_WHOLE=1   # a site's partial filled in one call
      # v2.3: short prompts; replies still identical to v1's fast mode
      export TF_GLM_PF_TAIL_MERGE=1   # a prompt's last 1-63 rows run in the chunk before them (no forward of their own)
      export TF_GLM_PROMPT_MICROBATCH_MIN_ROWS=960   # Mia's setting: chunks from 1,920 rows run in two halves (was 4,096)
      export TF_GLM_PF_BALANCE=1      # a prompt longer than a chunk in equal chunks, not full ones and a short last one
      # v2.4: 32-row verify windows, TensorFold 1.2.6's width (#2257) on this engine; replies identical
      export TF_GLM_MAX_ROWS=32       # a reply that repeats text it has seen (a file rewrite) takes up to 31 copied
      COPY_MAX="${COPY_MAX:-31}"      #   tokens a round, was 15: 252 -> 326 tok/s on a quoted file, 244 -> 316 on an edit
      WIDE_GRAPHS="${WIDE_GRAPHS:-16,32}"   # both widths replay CUDA graphs
      # exact mode keeps v1's settings: the same-bits prefill switches are not re-verified there yet (PART16 and ZC
      # only act on bf16 partials)
    fi
    ;;
  stock) ;;
  *) echo "vince/modes.sh: VINCE_MODE is fast, exact or stock, not '$VINCE_MODE'" >&2; return 1 ;;
esac
