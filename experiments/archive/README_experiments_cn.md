# Archive 分类说明（中文）

## 目录定位

- `v3/`
  - V3 主线历史入口与复现脚本。

- `compare/`
  - 横向对比脚本（baseline / layered_only / layered_fused）。

- `stagea/`
  - Stage-A 诊断与探针脚本（offset/gate/geo/pretrain audit）。

- `fusion/`（新增）
  - 按“交融模块”维度归类的脚本副本。
  - 当前包含：
    - `phase_fusion_compare_500.py`
    - `phase_fusion_layered_compare_500.py`
    - `phase_branch2_probe.py`

- `parallel/`（新增）
  - 按“并联/混合结构”维度归类的脚本副本。
  - 当前包含：
    - `phase_fusion_layered_v31_hybrid.py`

## 兼容性说明

- 为避免影响现有运行命令，`compare/` 与 `v3/` 原文件保留；
- `fusion/` 与 `parallel/` 目前是分类副本，便于阅读和管理。
