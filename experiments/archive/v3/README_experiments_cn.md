# 实验脚本分层说明（中文）

## 1. 目录职责

- `experiments/archive/v3/`
  - 作用：V3 主线历史入口与复现脚本。
  - 典型脚本：`phase_fusion_layered_v3_only.py`、`phase_fusion_layered_v31_hybrid.py`。

- `experiments/archive/compare/`
  - 作用：横向对比（baseline / layered_only / layered_fused）实验。
  - 典型脚本：`phase_fusion_layered_compare_500.py`、`phase_fusion_compare_500.py`、`eval_compare_1000.py`。

- `experiments/archive/stagea/`
  - 作用：Stage-A 诊断与验证（不是最终部署主线）。
  - 典型脚本：`stagea_diagnostics.py`、`stagea_geologit_probe.py`、`stagea_pretrain_audits.py`。

- `experiments/`（根目录）
  - 作用：当前活跃实验入口。
  - 典型脚本：`phase_fusion_scheme1.py`、`phase_fusion_stagea_schemeb.py`、`phaseadapter_v35_probe.py`、`train_pesq20_select_best.py`。

## 2. Stage-A 是什么

- Stage-A 是“离散 token 决策前的轻量校正阶段”。
- 常见形式：`offset + gate` 或几何约束 `GeoLogit`。
- 目标：在不改主干结构的前提下，验证 token 选择是否可改进。
- 结论：它更偏“诊断/验证阶段”，不等于最终稳定部署路径。

## 3. 你关心的 V3 配置链路

配置文件：
- `configs/phase_fusion_layered_v3_run100_match.yaml`

训练/评估入口：
- `experiments/archive/v3/phase_fusion_layered_v3_only.py`

底层复用函数：
- `experiments/archive/compare/phase_fusion_layered_compare_500.py`

关键权重字段说明：
- `baseline_cfg` / `baseline_ckpt`：baseline 参考模型。
- `layered_cfg` / `layered_ckpt`：离散主干微调后模型。
- `phase_cnn_ckpt`：相位分支。
- `init_adapter_ckpt`：融合器初始化权重。
- `adapter_ckpt`：本轮融合器训练输出权重。
- `report_json`：评估报告。

## 4. .yaml / .pt / .ckpt 区别

- `.yaml`：文本配置，可直接查看参数。
- `.pt`：PyTorch 权重文件，通常包含 `model`，有时也带 `config`。
- `.ckpt`：Lightning 检查点，通常包含 `state_dict`、优化器状态、训练步等更多信息。

如果 VSCode 直接打不开 `.pt/.ckpt`，可以用 Python 读取其键和值，再导出 json 文本。