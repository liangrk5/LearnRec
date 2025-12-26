from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_units: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for units in hidden_units:
            layers.extend([nn.Linear(prev, units), nn.ReLU(), nn.Dropout(dropout)])
            prev = units
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)


class EmbeddingLayer(nn.Module):
    def __init__(self, feature_sizes: Dict[str, int], embed_dim: int = 8):
        super().__init__()
        self.features = list(feature_sizes.keys())
        self.embedding = nn.ModuleDict(
            {name: nn.Embedding(size + 1, embed_dim) for name, size in feature_sizes.items()}
        )

    def forward(self, sparse_inputs: torch.Tensor) -> List[torch.Tensor]:
        embeds = []
        for idx, name in enumerate(self.features):
            embeds.append(self.embedding[name](sparse_inputs[:, idx]))
        return embeds


class MatrixFactorization(nn.Module):
    def __init__(self, feature_sizes: Dict[str, int], user_feature: str = "C1", item_feature: str = "C2", embed_dim: int = 16):
        super().__init__()
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.embeddings = nn.ModuleDict(
            {
                user_feature: nn.Embedding(feature_sizes[user_feature] + 1, embed_dim),
                item_feature: nn.Embedding(feature_sizes[item_feature] + 1, embed_dim),
            }
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, sparse_inputs: torch.Tensor):
        user_idx = sparse_inputs[:, 0]
        item_idx = sparse_inputs[:, 1]
        user_embed = self.embeddings[self.user_feature](user_idx)
        item_embed = self.embeddings[self.item_feature](item_idx)
        logits = (user_embed * item_embed).sum(dim=-1) + self.bias
        return logits


class FactorizationMachine(nn.Module):
    def __init__(self, feature_sizes: Dict[str, int], embed_dim: int = 8):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(feature_sizes, embed_dim)
        self.linear = nn.Linear(len(feature_sizes), 1)

    def forward(self, sparse_inputs: torch.Tensor):
        linear_part = self.linear(sparse_inputs.float()).squeeze(-1)
        embeddings = torch.stack(self.embedding_layer(sparse_inputs), dim=1)
        square_of_sum = embeddings.sum(dim=1) ** 2
        sum_of_square = (embeddings ** 2).sum(dim=1)
        interactions = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)
        return linear_part + interactions


class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, feature_sizes: Dict[str, int], embed_dim: int = 8):
        super().__init__()
        self.features = list(feature_sizes.keys())
        self.fields = len(self.features)
        self.embeddings = nn.ModuleDict()
        for i, feat in enumerate(self.features):
            for field in range(self.fields):
                self.embeddings[f"{feat}_to_{field}"] = nn.Embedding(feature_sizes[feat] + 1, embed_dim)

        self.linear = nn.Linear(len(feature_sizes), 1)

    def forward(self, sparse_inputs: torch.Tensor):
        linear_part = self.linear(sparse_inputs.float()).squeeze(-1)
        interactions = []
        for i in range(self.fields):
            for j in range(i + 1, self.fields):
                vi_fj = self.embeddings[f"{self.features[i]}_to_{j}"](sparse_inputs[:, i])
                vj_fi = self.embeddings[f"{self.features[j]}_to_{i}"](sparse_inputs[:, j])
                interactions.append((vi_fj * vj_fi).sum(dim=1))
        pairwise = torch.stack(interactions, dim=1).sum(dim=1)
        return linear_part + pairwise


class WideAndDeep(nn.Module):
    def __init__(self, feature_sizes: Dict[str, int], dense_dim: int, embed_dim: int = 8, hidden_units: Optional[List[int]] = None):
        super().__init__()
        hidden_units = hidden_units or [64, 32]
        self.embedding_layer = EmbeddingLayer(feature_sizes, embed_dim)
        input_dim = dense_dim + embed_dim * len(feature_sizes)
        self.deep = MLP(input_dim, hidden_units)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, dense_inputs: torch.Tensor, sparse_inputs: torch.Tensor):
        sparse_embeds = torch.cat(self.embedding_layer(sparse_inputs), dim=1)
        deep_out = self.deep(torch.cat([dense_inputs, sparse_embeds], dim=1))
        wide_linear = sparse_inputs.float().sum(dim=1)
        return deep_out + wide_linear + self.bias


class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, layer_num: int = 2):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim, 1)) for _ in range(layer_num)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(layer_num)])

    def forward(self, x0):
        x = x0
        for w, b in zip(self.weights, self.bias):
            xw = torch.matmul(x, w)  # (batch, 1)
            x = x0 * xw + b + x
        return x


class DeepCrossNetwork(nn.Module):
    def __init__(self, feature_sizes: Dict[str, int], dense_dim: int, embed_dim: int = 8, cross_layers: int = 2, hidden_units: Optional[List[int]] = None):
        super().__init__()
        hidden_units = hidden_units or [128, 64]
        self.embedding_layer = EmbeddingLayer(feature_sizes, embed_dim)
        input_dim = dense_dim + embed_dim * len(feature_sizes)
        self.cross_net = CrossNetwork(input_dim, layer_num=cross_layers)
        self.deep_net = MLP(input_dim, hidden_units)
        self.out = nn.Linear(input_dim + hidden_units[-1], 1)

    def forward(self, dense_inputs: torch.Tensor, sparse_inputs: torch.Tensor):
        sparse_embeds = torch.cat(self.embedding_layer(sparse_inputs), dim=1)
        x = torch.cat([dense_inputs, sparse_embeds], dim=1)
        cross = self.cross_net(x)
        deep = self.deep_net(x)
        return self.out(torch.cat([cross, deep], dim=1)).squeeze(-1)


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, hidden_units: Optional[List[int]] = None):
        super().__init__()
        hidden_units = hidden_units or [64, 16]
        layers = []
        prev = embed_dim * 2
        for units in hidden_units:
            layers.append(nn.Linear(prev, units))
            layers.append(nn.ReLU())
            prev = units
        layers.append(nn.Linear(prev, 1))
        self.attention = nn.Sequential(*layers)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor):
        query = query.unsqueeze(1).expand_as(keys)
        concat = torch.cat([query, keys], dim=-1)
        scores = self.attention(concat).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=1)
        weighted_sum = (weights.unsqueeze(-1) * keys).sum(dim=1)
        return weighted_sum


class DeepInterestNetwork(nn.Module):
    def __init__(self, feature_sizes: Dict[str, int], dense_dim: int, query_idx: int, history_indices: List[int], embed_dim: int = 8, hidden_units: Optional[List[int]] = None):
        super().__init__()
        hidden_units = hidden_units or [128, 64]
        self.features = list(feature_sizes.keys())
        self.query_idx = query_idx
        self.history_indices = history_indices
        self.embedding_layer = EmbeddingLayer(feature_sizes, embed_dim)
        self.attention = AttentionPooling(embed_dim)
        input_dim = dense_dim + embed_dim * (len(feature_sizes) + 1)
        self.dnn = MLP(input_dim, hidden_units)

    def forward(self, dense_inputs: torch.Tensor, sparse_inputs: torch.Tensor, history: torch.Tensor, history_length: torch.Tensor):
        embeddings = self.embedding_layer(sparse_inputs)
        query_embed = embeddings[self.query_idx]
        history_embeds = []
        for idx in self.history_indices:
            history_embeds.append(self.embedding_layer.embedding[self.features[idx]](history[:, idx - self.history_indices[0]]))
        history_stack = torch.stack(history_embeds, dim=1)
        mask = torch.arange(history_stack.size(1), device=history_stack.device).unsqueeze(0) < history_length.unsqueeze(1)
        pooled_history = self.attention(query_embed, history_stack, mask)
        concat = torch.cat([dense_inputs, torch.cat(embeddings, dim=1), pooled_history], dim=1)
        return self.dnn(concat)


def build_model(name: str, feature_info: Dict, embed_dim: int = 8):
    name = name.lower()
    feature_sizes = feature_info["feature_sizes"]
    dense_dim = len(feature_info["dense_features"])

    if name == "mf":
        return MatrixFactorization(feature_sizes, feature_info["query_feature"], feature_info["history_features"][0], embed_dim)
    if name == "fm":
        return FactorizationMachine(feature_sizes, embed_dim)
    if name == "ffm":
        return FieldAwareFactorizationMachine(feature_sizes, embed_dim)
    if name == "wdl":
        return WideAndDeep(feature_sizes, dense_dim, embed_dim)
    if name == "dcn":
        return DeepCrossNetwork(feature_sizes, dense_dim, embed_dim)
    if name == "din":
        query_idx = feature_info["sparse_features"].index(feature_info["query_feature"])
        history_indices = [feature_info["sparse_features"].index(f) for f in feature_info["history_features"]]
        return DeepInterestNetwork(feature_sizes, dense_dim, query_idx, history_indices, embed_dim)
    raise ValueError(f"Unknown model name: {name}")
