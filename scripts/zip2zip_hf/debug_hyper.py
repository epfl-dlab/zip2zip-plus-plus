import torch
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer

EXPORT_DIR = "/mnt/scratch/export/zip2zip_1b_step1908"

model = Zip2ZipModel.from_pretrained(
    EXPORT_DIR,
    dtype=torch.bfloat16,
).to("cuda").eval()

tokenizer = Zip2ZipTokenizer.from_pretrained(EXPORT_DIR)

initial_vocab_size = model.zip2zip_config.compression.initial_vocab_size
max_codebook_size = model.zip2zip_config.compression.max_codebook_size
print(f"initial_vocab_size={initial_vocab_size}, max_codebook_size={max_codebook_size}")

prompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
inputs = tokenizer([prompt], return_tensors="pt", padding="longest").to("cuda")
input_ids = inputs["input_ids"]
print(f"\nprompt tokens: {input_ids.shape[1]}")
print(f"token ids: {input_ids[0].tolist()}")

model.codebook_manager.init_codebooks_and_hyper_weight_cache(batch_size=1)

with torch.no_grad():
    outputs = model.base_model(input_ids=input_ids, return_dict=True)

logits = outputs.logits[0, -1]  # last position
print(f"\nlogits shape: {logits.shape}")
print(f"  base logits [0:{initial_vocab_size}]: min={logits[:initial_vocab_size].min():.2f}, max={logits[:initial_vocab_size].max():.2f}")

hyper_start = initial_vocab_size
hyper_end = initial_vocab_size + max_codebook_size
hyper_logits = logits[hyper_start:hyper_end]
print(f"  hyper logits [{hyper_start}:{hyper_end}]: min={hyper_logits.min():.2f}, max={hyper_logits.max():.2f}")

special_logits = logits[hyper_end:]
print(f"  special logits [{hyper_end}:]: min={special_logits.min():.2f}, max={special_logits.max():.2f}")

top_base = logits[:initial_vocab_size].topk(5)
print(f"\ntop-5 base tokens:")
for i, (val, idx) in enumerate(zip(top_base.values, top_base.indices)):
    print(f"  {i}: id={idx.item()} ({tokenizer.tokenizer.decode([idx.item()])!r}) logit={val.item():.2f}")

top_hyper = hyper_logits.topk(5)
print(f"\ntop-5 hyper tokens:")
for i, (val, idx) in enumerate(zip(top_hyper.values, top_hyper.indices)):
    print(f"  {i}: codebook_idx={idx.item()} logit={val.item():.2f}")

top_all = logits.topk(10)
print(f"\ntop-10 overall:")
for i, (val, idx) in enumerate(zip(top_all.values, top_all.indices)):
    region = "base" if idx < initial_vocab_size else ("hyper" if idx < hyper_end else "special")
    print(f"  {i}: id={idx.item()} region={region} logit={val.item():.2f}")

cb_updates = model.codebook_manager.updates
cb_indices = model.codebook_manager.updates_indices
if cb_updates is not None:
    print(f"\ncodebook updates shape: {cb_updates.shape}")
    print(f"codebook indices (per batch): {[len(ui) for ui in cb_indices]}")
    print(f"first few indices: {cb_indices[0][:10]}")
else:
    print("\nno codebook updates!")

model.codebook_manager.reset()
