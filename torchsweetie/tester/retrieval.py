from pathlib import Path

import pandas as pd
import torch
from rich import print
from torch import Tensor
from tqdm import tqdm

from ..data import create_cls_dataloader
from ..utils import DIR_B, DIR_E, MODELS, SIMILARITY, get_config


class RetrievalTester:
    NCOLS = 100

    def __init__(self, cfg_file: str, run_dir: str, exp_dir: str, weights: str) -> None:
        # Get the root path (project path)
        self.root_dir = Path.cwd()

        # Get the absolute path of config file and load it
        self.cfg_file = self.root_dir / cfg_file
        self.cfg = get_config(self.root_dir, self.cfg_file)

        # Running directory, used to record results and models
        self.exp_dir = self.root_dir / run_dir / self.cfg_file.stem / exp_dir
        assert self.exp_dir.exists()
        print(f"Experimental directory: {DIR_B}{self.exp_dir}{DIR_E}")

        # Model
        model_weights = self.exp_dir / weights
        self.cfg.model.weights = model_weights
        self.model = MODELS.create(self.cfg.model)
        self.model.cuda()

        # P.S. Loss function should be dropped

        # Dataloader
        dataloader_cfg = self.cfg.test_dataloader
        self.dataloader = create_cls_dataloader(dataloader_cfg)

        # Target names
        target_names = dataloader_cfg.dataset.target_names
        self.target_names = pd.read_csv(target_names, header=None)[0].to_list()

        self.similarity_fn = SIMILARITY.create(self.cfg.similarity)

        # Store the embeddings output by model
        self.embeddings: Tensor
        self.labels: Tensor

    @torch.no_grad()
    def test(self) -> None:
        pbar = tqdm(desc=f"Testing", total=len(self.dataloader), ncols=self.NCOLS)

        self.model.eval()

        embeddings_list = []
        labels_list = []

        for images, labels in self.dataloader:
            labels_list.append(labels)
            images, labels = images.cuda(), labels.cuda()
            outputs = self.model(images)  # (B, N)
            embeddings_list.append(outputs)

            pbar.update()

        pbar.close()

        self.embeddings = torch.concat(embeddings_list)
        self.labels = torch.concat(labels_list)

    @torch.no_grad()
    def report(self, embeddings: Tensor, labels: Tensor, topk_list: list[int], digits: int) -> None:
        # (B, N) & (N, R) -> (B, R) -> (B, K)
        similarity = self.similarity_fn(self.embeddings, embeddings)
        _, indices = similarity.topk(max(topk_list), 1)
        indices = indices.cpu()

        # 计算最长类名
        W = 0
        for name in self.target_names:
            length = self._display_len(name)
            W = max(W, length)

        # Recall@K
        header = f"\n{'':>{W}}"
        for i, k in enumerate(topk_list):
            header += f"{f'Recall@{k}':>{12}}"
        header += "\n"
        print(header)

        D = digits

        recall_metrics = {}
        for name in self.target_names:
            recall_metrics[name] = {}
            for k in topk_list:
                recall_metrics[name][k] = []

        # (B, K) == (B, 1) -> (B, K), where K is like [True, False, ...]
        recall = labels[indices] == self.labels[:, None]
        for i in range(len(self.labels)):
            name = self.target_names[self.labels[i]]
            for k in topk_list:
                recall_metrics[name][k].append(True in recall[i, :k])

        micro_avg = {k: [] for k in topk_list}

        for name in self.target_names:
            line = self._format_string(name, W)
            recall = recall_metrics[name]
            for k in topk_list:
                recall_k = recall[k]
                recall_k = sum(recall_k) / len(recall_k)
                micro_avg[k].append(recall_k)
                line += f"{recall_k:>12.{D}f}"
            print(line)

        line = f"\n{'average':>{W}}"
        for k in topk_list:
            average = sum(micro_avg[k]) / len(micro_avg[k])
            line += f"{average:>12.{D}f}"
        print(line)

    @staticmethod
    def _is_chinese(c: str) -> bool:
        assert len(c) == 1

        if "\u4e00" <= c <= "\u9fa5":
            return True
        else:
            return False

    def _display_len(self, string: str) -> int:
        length = 0
        for c in string:
            if self._is_chinese(c):
                length += 2
            else:
                length += 1

        return length

    def _format_string(self, string: str, max_length: int) -> str:
        length = self._display_len(string)

        while length < max_length:
            string = " " + string
            length += 1

        return string
