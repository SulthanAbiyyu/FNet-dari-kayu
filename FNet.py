import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_position):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_position, embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        seq_len = input_ids.size(-1)
        position_ids = torch.arange(seq_len, dtype=torch.long).view(1, -1)

        token_embed = self.token_embeddings(input_ids)
        position_embed = self.position_embeddings(position_ids)

        embeddings = token_embed + position_embed
        embeddings = self.layernorm(embeddings)
        return self.dropout(embeddings)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_dim),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.layernorm(x)
        return self.ff(x)


class FNetSublayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.layernorm(x)
        f_h = torch.fft.fft(x, dim=-1)  # embedding dimension
        f_seq = torch.fft.fft(f_h, dim=-2)  # sequence dimension
        return f_seq.real  # "we only keep the real part of the result"


class FNet(nn.Module):
    def __init__(self, n_layers, embed_dim, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList(
                [
                    FNetSublayer(embed_dim),
                    FeedForward(embed_dim, hidden_size)
                ]) for _ in range(n_layers)
        ])

    def forward(self, x):
        for fnet, ff in self.layers:
            x = x + fnet(x)  # residual connection
            x = x + ff(x)

        return x


if __name__ == "__main__":
    dummy_data = torch.randn(10, 100, 768)  # Batch x Seq_len x embed_dim
    fnet = FNet(4, 768, 768 * 2)

    out = fnet(dummy_data)
    print(out.shape)
