"""CPU integration check with tiny random weights; not a benchmark."""
import json
import tempfile

import torch
from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion

from bdlm_serving.runtime.diffusiongemma import account, observe


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = DiffusionGemmaConfig(
        canvas_length=4,
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=4, num_key_value_heads=4, head_dim=8),
        text_config=dict(vocab_size=128, hidden_size=32, num_attention_heads=4,
                         num_key_value_heads=2, num_global_key_value_heads=2, head_dim=8,
                         global_head_dim=8, num_experts=4, top_k_experts=2,
                         moe_intermediate_size=32, num_hidden_layers=2, intermediate_size=64,
                         layer_types=["sliding_attention", "full_attention"], max_position_embeddings=64))
    model = DiffusionGemmaForBlockDiffusion(config).eval()
    with tempfile.TemporaryDirectory() as folder:
        model.save_pretrained(folder)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            folder, device_map={"": "cpu"}, dtype=torch.bfloat16, attn_implementation="sdpa",
            experts_implementation="grouped_mm", local_files_only=True).eval()
    assert model.model.encoder.language_model.embed_tokens.weight is model.model.decoder.embed_tokens.weight
    json.dumps(model.generation_config.to_dict())
    prompt = torch.tensor([[2, 3, 4]])
    # Out-of-vocabulary EOS forces crossing one boundary on random weights.
    with torch.inference_mode(), observe(model) as events:
        out = model.generate(prompt, max_new_tokens=8, max_denoising_steps=2,
                             eos_token_id=[999], pad_token_id=0)
    _, metrics = account(events, out.sequences[0, 3:].tolist(), 3, 4, 8, {999}, {0}, 0,
                         out.tokens_per_forward.item())
    assert metrics["reencode_forwards"] == 1 and metrics["truncated"]
    print("tiny native trace:", events)


if __name__ == "__main__":
    main()
