"""
inference_utils.py — Shared model loading and generation.
=========================================================

Both run_turn1.py and run_turn2.py import from here. torch / transformers are
imported lazily so --dry-run works without a GPU.
"""

from __future__ import annotations

from datetime import datetime

import config


def load_model_and_tokenizer():
    import torch                                                    # noqa
    from transformers import AutoModelForCausalLM, AutoTokenizer    # noqa

    dtype = getattr(torch, config.TORCH_DTYPE)
    print(f"[{datetime.now():%H:%M:%S}] Loading tokenizer: {config.MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    print(f"[{datetime.now():%H:%M:%S}] Loading model "
          f"(dtype={config.TORCH_DTYPE}, device_map={config.DEVICE_MAP}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, torch_dtype=dtype,
        device_map=config.DEVICE_MAP, trust_remote_code=True,
    )
    model.eval()
    print(f"[{datetime.now():%H:%M:%S}] Model loaded.")
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict],
             max_new_tokens: int, seed: int) -> str:
    import torch                          # noqa
    from transformers import set_seed     # noqa

    set_seed(seed)
    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)
        attention_mask = torch.ones_like(input_ids)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=config.DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )
        if config.DO_SAMPLE:
            gen_kwargs.update(temperature=config.TEMPERATURE, top_p=config.TOP_P)
        output_ids = model.generate(input_ids, **gen_kwargs)
        new_tokens = output_ids[0, input_ids.shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()