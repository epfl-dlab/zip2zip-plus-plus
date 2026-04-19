from zip2zip_core.codebook import CodebookManager
import torch




if __name__ == "__main__":
    # Example usage of CodebookManager
    initial_vocab_size = 128000
    max_codebook_size = 4096
    max_subtokens = 4
    embedding_dim = 512
    pad_token_id = 128001

    codebook_manager = CodebookManager(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=max_codebook_size,
        max_subtokens=max_subtokens,
        embedding_dim=embedding_dim,
        pad_token_id=pad_token_id,
        disabled_ids=[pad_token_id]
    )

    # Simulate some token IDs to update the codebooks with
    prompt_token_ids = torch.randint(0, initial_vocab_size, (2, 8))  # batch of 4 sequences, each of length 128
    
    codebook_manager.update_codebooks(prompt_token_ids)

    # self.updates = None
    # self.updates_indices = None

    print("CodebookManager.updates:", codebook_manager.updates)
    print("CodebookManager.updates_indices:", codebook_manager.updates_indices)

    # simulate decoding a sequence of codebook token IDs
    steps = 3
    for i in range(steps):
        output_token_ids = torch.randint(0, initial_vocab_size, (2, 1))  # batch of 2 sequences, each of length 1
        codebook_manager.update_codebooks(output_token_ids)
        print(f"After step {i+1}:")
        print("  CodebookManager.updates:", codebook_manager.updates)
        print("  CodebookManager.updates_indices:", codebook_manager.updates_indices)


"""

  CodebookManager.updates_indices: [[17], [17], [17], [17]]
root@bolt-6fbi8gpwhm-w88gaatwas ➜  zip2zip-core git:(main) ✗ python scripts/test_codebook.py
CodebookManager.updates: tensor([[[ 51614,  24929, 128001, 128001],
         [ 24929, 111353, 128001, 128001],
         [111353,  55719, 128001, 128001],
         ...,
         [128001, 128001, 128001, 128001],
         [128001, 128001, 128001, 128001],
         [128001, 128001, 128001, 128001]],

        [[ 99358,  68252, 128001, 128001],
         [ 68252,  74234, 128001, 128001],
         [ 74234,  24416, 128001, 128001],
         ...,
         [128001, 128001, 128001, 128001],
         [128001, 128001, 128001, 128001],
         [128001, 128001, 128001, 128001]]])
CodebookManager.updates_indices: [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]
After step 1:
  CodebookManager.updates: tensor([[[ 84918,  16742, 128001, 128001]],

        [[ 79214,   7938, 128001, 128001]]])
  CodebookManager.updates_indices: [[7], [7]]
After step 2:
  CodebookManager.updates: tensor([[[ 16742,  45368, 128001, 128001]],

        [[  7938, 125619, 128001, 128001]]])
  CodebookManager.updates_indices: [[8], [8]]
After step 3:
  CodebookManager.updates: tensor([[[ 45368,   3726, 128001, 128001]],

        [[125619,  11251, 128001, 128001]]])
  CodebookManager.updates_indices: [[9], [9]]





"""