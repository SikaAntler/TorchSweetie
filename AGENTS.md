# 仓库指南

## 项目结构与模块组织
TorchSweetie 是面向图像分类、图像检索和目标检测的轻量级 PyTorch 框架。
核心代码位于 `torchsweetie/`：
  - `data/` 管理数据集、变换、采样器和指标；
  - `models/` 包含分类模型和检测器组件；
  - `engine/` 实现训练、评估和导出流程；
  - 辅助模块包括 `losses/`、`optim/`、`retrieval/` 和 `utils/`。
命令行入口位于 `tools/`。
仓库尚未纳入自动化测试、示例配置或数据集资源。

## 构建、测试与开发命令
在仓库根目录执行以下命令，并将示例路径替换为实际路径：
  - 安装：`pip install -e .` 以可编辑模式安装项目及依赖；
  - 训练：`python tools/train.py --task classification --cfg path/to/config.yaml` 启动分类训练，目标检测任务使用 `--task detection`；
  - 测试：`python tools/test.py --task classification --exp path/to/experiment --best --cfg path/to/config.yaml` 评估已保存的最佳检查点；
  - 导出：`python tools/export.py --help` 查看模型导出选项。

## 编码风格与命名约定
遵循 `pyproject.toml` 中 Black 和 Ruff 的配置。
沿用现有类型注解和现代 Python 语法。
函数和库模块采用 `snake_case`，类名采用 `PascalCase`，注册表常量采用全大写名称。
多词命令行脚本使用连字符，例如 `k-fold-cross-validate.py`。
可配置组件应通过 `torchsweetie/utils/registry.py` 中对应的注册表注册。

## 测试指南
项目尚未配置自动化测试框架或覆盖率门槛。
`tools/test.py` 用于模型评估，并非单元测试入口。
