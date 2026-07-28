from torch import nn

from ...data import ClsDataPack, ClsModelOutput


class ClsModel(nn.Module):
    def __init__(
        self, backbone: nn.Module, head: nn.Module, requires_data_pack: bool = False
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.requires_data_pack = requires_data_pack

    def forward(self, data: ClsDataPack) -> ClsModelOutput:
        embeddings = self.backbone(data if self.requires_data_pack else data.inputs)
        logits = self.head(embeddings)

        return ClsModelOutput(logits, embeddings)
