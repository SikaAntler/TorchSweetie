from torch import Tensor, nn

from ..utils import SIMILARITY

__all__ = [
    "ChebyshevDistance",
    "EuclideanDistance",
    "ManhattanDistance",
    "CosineSimilarity",
]


@SIMILARITY.register()
class ChebyshevDistance(nn.Module):

    def forward(self, embed1: Tensor, embed2: Tensor) -> Tensor:
        # (B1, 1, N) - (1, B2, N) -> (B1, B2, N) -> (B1, B2)
        similarity, _ = (embed1[:, None] - embed2[None]).abs().max(dim=2)

        return similarity


@SIMILARITY.register()
class EuclideanDistance(nn.Module):

    def forward(self, embed1: Tensor, embed2: Tensor) -> Tensor:
        # (B1, N) @ (B2, N) -> (B1, B2)
        similarity = embed1 @ embed2.T

        return similarity


@SIMILARITY.register()
class ManhattanDistance(nn.Module):

    def forward(self, embed1: Tensor, embed2: Tensor) -> Tensor:
        # (B1, 1, N) - (1, B2, N) -> (B1, B2, N) -> (B1, B2)
        similarity = (embed1[:, None] - embed2[None]).abs().sum(dim=2)

        return similarity


@SIMILARITY.register()
class CosineSimilarity(nn.Module):

    def forward(self, embed1: Tensor, embed2: Tensor) -> Tensor:
        embed1 /= embed1.norm(2, 1, True)
        embed2 /= embed2.norm(2, 1, True)

        # (B1, N) @ (N, B2) -> (B1, B2)
        similarity = embed1 @ embed2.T

        return similarity
