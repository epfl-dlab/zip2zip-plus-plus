import torch
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer

EXPORT_DIR = "/mnt/scratch/export/zip2zip_1b_step1908"

model = Zip2ZipModel.from_pretrained(
    EXPORT_DIR,
    dtype=torch.bfloat16,
).to("cuda").eval()

tokenizer = Zip2ZipTokenizer.from_pretrained(EXPORT_DIR)

prompts = [
    "Please write a MultiHeadAttention layer in PyTorch.",
]

inputs = tokenizer(prompts, return_tensors="pt", padding="longest").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=128,
        use_cache=True,
        top_k=50,
    )

for text in tokenizer.batch_decode(outputs, skip_special_tokens=True):
    print(text)
    print("=" * 40)

print("\n--- Color decode ---\n")
for text in tokenizer.color_decode(outputs, color_scheme="finegrained"):
    print(text)
    print("=" * 40)