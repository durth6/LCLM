import torch

from latent_context.selector import CosineSelector, LatentMemory


class CountingEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode_once(self) -> LatentMemory:
        self.calls += 1
        return LatentMemory(
            latents=torch.eye(4),
            latent_start=torch.tensor([0, 8, 16, 24], dtype=torch.int32),
            latent_end=torch.tensor([8, 16, 24, 32], dtype=torch.int32),
            latent_index=torch.arange(4, dtype=torch.int32),
            segment_id=torch.zeros(4, dtype=torch.int16),
            window_id=torch.zeros(4, dtype=torch.int32),
            fingerprint="unit-test",
        )


def test_one_memory_encoding_is_reused_for_multiple_queries():
    encoder = CountingEncoder()
    memory = encoder.encode_once()
    selector = CosineSelector()
    queries = torch.eye(4)[:3]
    for query in queries:
        output = selector(query.unsqueeze(0), memory, k=1)
        assert output.indices.shape == (1, 1)
    assert encoder.calls == 1
