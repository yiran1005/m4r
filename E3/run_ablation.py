# -*- coding: utf-8 -*-
"""
Run the plan-components ablation inference with vLLM.

For each variant it builds prompts from S_full_14B, generates ONE completion per
sample (greedy: temperature=0, top_p=1, fixed seed -- no majority vote), and
writes raw outputs to OUTPUT_DIR/raw/{model}_{variant}_raw.csv.

"""
import os
import argparse
import pandas as pd

import config
import prompts as P


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(config.MODELS.keys()))
    ap.add_argument("--variants", nargs="*", default=None,
                    help="subset of variants; default = config.MODELS[model]['variants']")
    ap.add_argument("--data", default=config.DATA_PATH)
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N rows")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="feed prompts as raw completions instead of the chat template")
    return ap.parse_args()


def load_model(model_cfg):
    from vllm import LLM
    from transformers import AutoTokenizer
    tp = model_cfg.get("tensor_parallel_size", 1)
    tok = AutoTokenizer.from_pretrained(model_cfg["model_path"], trust_remote_code=True)
    kwargs = dict(
        model=model_cfg["model_path"],
        tensor_parallel_size=tp,
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.90),
        max_model_len=config.MAX_MODEL_LEN,
        trust_remote_code=True,
        seed=config.SEED,
    )
    # For single-GPU, force the multiprocessing backend so vLLM does NOT
    # initialise a Ray cluster (some vLLM versions otherwise spin up Ray and
    # then miscount GPUs -> "required GPUs exceeds available GPUs"). Ray is only
    # needed for genuine multi-GPU tensor parallelism (tp > 1).
    if tp == 1:
        kwargs["distributed_executor_backend"] = "mp"
    try:
        llm = LLM(**kwargs)
    except TypeError:
        # older vLLM may not accept distributed_executor_backend -> drop it
        kwargs.pop("distributed_executor_backend", None)
        llm = LLM(**kwargs)
    return llm, tok


def sampling_params():
    from vllm import SamplingParams
    return SamplingParams(
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        max_tokens=config.MAX_NEW_TOKENS,
        seed=config.SEED,
    )


def render(tok, messages, use_chat_template):
    if use_chat_template:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # raw completion: concatenate message contents
    return "\n".join(m["content"] for m in messages)


def generate(llm, tok, conversations, use_chat_template):
    """conversations: list of message-lists. Returns list of generated strings."""
    sp = sampling_params()
    rendered = [render(tok, c, use_chat_template) for c in conversations]
    outs = llm.generate(rendered, sp)
    # vLLM preserves input order
    return [o.outputs[0].text for o in outs]


def run_single_turn(llm, tok, df, variant, use_ct):
    conversations = [P.build_messages(row, variant) for _, row in df.iterrows()]
    outputs = generate(llm, tok, conversations, use_ct)
    res = df.copy()
    res["variant"] = variant
    res["model_output"] = outputs
    res["turn1_output"] = ""
    return res


def run_v4(llm, tok, df, use_ct):
    # Turn 1: zero-shot CoT
    t1_conv = [P.build_v4_turn1_messages(row) for _, row in df.iterrows()]
    t1_out = generate(llm, tok, t1_conv, use_ct)
    # Turn 2: post-hoc self-check
    t2_conv = [P.build_v4_turn2_messages(row, o) for (_, row), o in zip(df.iterrows(), t1_out)]
    t2_out = generate(llm, tok, t2_conv, use_ct)
    res = df.copy()
    res["variant"] = "V4"
    res["turn1_output"] = t1_out
    res["model_output"] = t2_out          # final = post-self-check answer
    return res


def main():
    args = parse_args()
    model_cfg = config.MODELS[args.model]
    variants = args.variants or model_cfg["variants"]
    use_ct = not args.no_chat_template

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit).copy()
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    print(f"[i] Loaded {len(df)} samples from {args.data}")

    os.makedirs(os.path.join(config.OUTPUT_DIR, "raw"), exist_ok=True)

    llm, tok = load_model(model_cfg)

    keep_cols = ["row_id", "dataset", "refined_question", "relevant_column",
                 "task", "difficulty", "results", "prompt"]
    base = df[keep_cols].copy()

    for variant in variants:
        out_path = config.raw_path(args.model, variant)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_path} exists (use --overwrite)")
            continue
        print(f"[run ] {args.model} / {variant} ...")
        if variant in config.TWO_TURN_VARIANTS:
            res = run_v4(llm, tok, base, use_ct)
        else:
            res = run_single_turn(llm, tok, base, variant, use_ct)
        res.to_csv(out_path, index=False)
        print(f"[save] {out_path}  ({len(res)} rows)")

    print("[done] inference complete.")


if __name__ == "__main__":
    main()