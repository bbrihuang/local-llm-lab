from transformers import AutoConfig, AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"
cfg = AutoConfig.from_pretrained(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)

head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
print("Layers (transformer blocks):", cfg.num_hidden_layers)
print("Hidden size (numbers per token):", cfg.hidden_size)
print("Attention heads:", cfg.num_attention_heads)
print("Key/value heads:", cfg.num_key_value_heads)
print("Head size:", head_dim)
print("MLP size:", cfg.intermediate_size)
print("Vocabulary size:", cfg.vocab_size)
print("Max context (tokens):", cfg.max_position_embeddings)

# KV cache: for every token, every layer stores one key and one value per KV head, 2 bytes each
kv_bytes = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2
print(f"\nKV cache per token: {kv_bytes / 1024:.0f} KB")
print(f"KV cache at max context: {kv_bytes * cfg.max_position_embeddings / 2**20:,.0f} MiB")
print(f"KV cache at 8,192 tokens: {kv_bytes * 8192 / 2**20:,.0f} MiB")

text = "The quick brown fox jumps over the lazy dog."
ids = tok(text)["input_ids"]
print(f"\n'{text}' -> {len(ids)} tokens")
print([tok.decode([i]) for i in ids])
print(ids)