# 论文与开源代码对应核对（2026-10-19）

本目录基于原 GitHub 仓库 `main` 的 `e8063eb67a497b35a8f5b8f96a26a19457bb38a1` 整理，并核对了最新历史汇总分支 `sync/paper-artifacts-20260615`（`df60ca2`）。对应论文主文件和各 Section 的 SHA256 记录在 `provenance/source_manifest.json`。现统一保存在独立私有仓库 `HKUJasonJiang/Seismic-FNO-Reproducibility` 的 `main` 分支；论文、回复信和旧仓库均未改动。

## 已整理并验证的内容

- 原基准、数据稀缺、PGA/采样率外推、六结构、P-58/FRS/持续时间，以及 E1–E5 补充实验的相关计算代码与配置。
- 六个架构共 54 个完整筛选记录，每个 20 epoch；6 个最终完整训练记录，每个 50 epoch。保留配置、逐 epoch 日志、结果与原始来源哈希，去除服务器调度、重复镜像和测试夹具。
- 冻结 GM 分组划分与完整测试集。验证训练/验证/测试 GM 无交集、候选选择只使用验证 MSE，最终汇总与六个单独结果一致。
- 从公开数据集的固定版本补齐原 O1/O3 训练日志，并保存数据和原模型的下载地址、大小、SHA256。模型大文件不直接塞入 Git。
- 从 F10/FW10 各 27,018 条记录的完整 90 点 FRS 重新计算 E2，1.0 g 原表数值吻合（误差 < 1e-10 个百分点）；边界扫描一致率 94.02%–98.29%。

## 1. 六架构调参需要补入正文，原来的“最多提升 0.005”只适用于前三类

当前 §5.1 的新增段落仍只列 FNO、Transformer、BiLSTM。完整结果如下，原值取原固定配置表，新增值取验证集选择后的完整训练结果：

| 架构 | 原配置 R² | 调参后 R² | 增量 | 实数参数量倍率 |
|---|---:|---:|---:|---:|
| BiLSTM | 0.987250 | 0.987809 | 0.000559 | 1.749 |
| Transformer | 0.972503 | 0.975150 | 0.002646 | 1.397 |
| FNO | 0.942307 | 0.947302 | 0.004994 | 4.000 |
| ResCNN1D | 0.809627 | 0.870106 | 0.060479 | 1.500 |
| UNet1D | 0.722664 | 0.780096 | 0.057432 | 4.008 |
| MLP | 0.046200 | 0.387150 | 0.340949 | 1.777 |

可支持的结论是：**六类的排序没有改变；前三类改进均不超过 0.005，后三类有较明显提升，但仍低于 FNO。** 不宜把“全部六类提升都不超过 0.005”写入正文，也不宜用这些调参结果替换原表而保留原配置和训练成本。

建议替换段落（供作者审阅，不自动写入 TeX）：

> To assess the sensitivity of the comparison to configuration choice, nine feasible candidate configurations were evaluated for each of the six architectures. Candidates were screened for 20 epochs using the same 19 amplitude levels and grouped training/validation split, and the candidate with the lowest validation MSE in each family was then trained from initialization for 50 epochs using all 57 levels. The selected configurations retained the accuracy ordering BiLSTM, Transformer, FNO, ResCNN1D, UNet1D, and MLP. The test R² improvements for BiLSTM, Transformer, and FNO were 0.0006, 0.0026, and 0.0050, respectively. ResCNN1D, UNet1D, and MLP improved to R² values of 0.8701, 0.7801, and 0.3871, respectively, and remained below FNO. These results support the robustness of the observed accuracy ordering to the evaluated configuration choices. The original configurations are retained for the subsequent comparisons to preserve the benchmark protocol.

这是相同候选数、epoch 数、数据划分和数值精度的检查，不能表述成相同 FLOPs、相同 GPU 小时或穷尽式最优调参。新增三类已经整理到独立 E5 扩展目录，并标明其正文整合状态。

## 2. 参数量口径需要说清楚

原 Table 3 中 FNO-Large 为 8,505,793 个 `numel`（复数元素计一个），按实数标量计为 16,927,169。其他架构均为实参数。两种数值都保留，E1/E5 使用 `real_params`。

建议在表注说明计数规则，并将 “parameter-matched” 限定为近似同量级配置比较；不能把复数元素计数与实数参数计数混用来论证严格等参数预算。采用新的计数口径不需要更换原模型或重算其性能。FNO 调参后为 34,018,177 个元素 / 67,703,681 个实数标量。

## 3. O4 与 O5 默认值已按实际归档修正

- 六结构 O4 归档配置全为 **batch 1024**，旧脚本默认是 1536；公开入口默认已改为 1024，命令仍明确写出。它不同于 O1 六模型比较的 batch 64。正文若说明训练设置，应分开写。
- O5 历史 period config 写 120，但用于当前论文和 E2 的全量 FRS 实际是 **90 个周期点**。公开 FRS 入口默认和复现命令均采用 90，同时保留原配置字节及差异说明。
- FRS 与设备筛选默认改为当前正文采用的 supported 三个周期带。旧的额外代理周期带不纳入公开实验配方。
- F-6 单结构数据目录显式隔离，避免读入六结构全部 HDF5 导致训练样本数和任务改变。
- 旧 `placement_supported` 输出使用了另一套周期网格，与当前论文的 90 点 `frs_full` 不同；本次未将旧输出混作论文结果纳入公开包。当前设备筛选结果由完整 FRS 重新计算，正文数值以 `chapter6_revision` 与 E2 的全量复算为准。P-58 的三个汇总 CSV 与原归档逐项吻合。

这些是执行入口与实际实验的对齐，没有改变已存结果或重新选择实验结论。

## 4. 采样率、RoPE 与推理成本的解释边界

当前 RoPE 段落的数值与 E4 一致：20 Hz matched R²=0.9729，50 Hz direct-index R²=-0.4818。固定 15 点 patch 在不同采样率下对应不同物理时长，不能据此概括所有 Transformer。

E4 同时包含输入/输出重采样 wrapper 控制。例如原 20 Hz Transformer 的 direct-50 R²=-0.5242，而 wrapped-50 为 0.9613；它们是不同部署流程。保留完整控制结果便于读者评估“直接迁移”的优势。正文应一直明确 direct 推理，与 wrapper 区分。

Table 4 的 Transformer R²=0.9725 与 Table 7 的 50 Hz Transformer R²=0.9727 是不同训练运行，不能合并或交叉使用权重。

E1 的 23× 推理延迟和 12×显存比较来自 RTX 5090、batch 64；测量采用随机初始化以比较架构成本。它不是训练速度比较，也不是由这组随机权重计算准确率。原训练成本和新推理成本分别保留。

## 5. 敏感性结论应对应实际扫描和聚合方式

当前 §6.2 已报告 94–98% 的 band-threshold 一致率，并给出 0.4 g 下 F-10 刚性设备的分类差异；这些有计算支持。建议补充扫描步长：容量 0.05 g，下/上分类边界均 1 个百分点。

每组边界的一致率分母是 **6 个结构-周期带 × 39 个容量 = 234 个案例**。它不是“每个容量处六个带均完全一致”：在 1.0 g 下，扫描的边界组合中最低为 4/6 个带一致。储物架最大差异虽仅 1.159 个百分点，足够贴近边界时仍可能改变类别；“对边界不敏感”应理解为本次总体扫描中的相对稳定，而非任意边界的严格不变。

代码定义超越为 `Sa >= capacity`，类别为 `<10%`、`[10%,50%)`、`>=50%`。正文中若使用 “exceeds 50%”，建议与等号约定对齐。旧结果文件名中的 `table11` 指设备筛选表，当前正文编号是 Table 12；映射以 `tab:o5_frs_matrix` 为准。

## 6. 尚不能称为完整的一键端到端复现

| 尚需补齐 | 当前证据与影响 | 建议处理 |
|---|---|---|
| E4 RoPE 和 E5 六类最终 checkpoint 的公开地址及内容哈希 | 训练完成、日志和指标齐全；公开数据/模型清单未发现这些新增权重 | 从完成训练的服务器归档 best checkpoint，保存 SHA256，再上传模型托管或 release assets；不把大权重直接提交普通 Git |
| 完整 O2 run-level 日志/权重 | 完整论文指标表在本目录；已发现的公开 O2 档案仅旧 partial E-AC80 子集 | 收齐 Fig.13/Table8 对应每个运行的配置、日志和 best checkpoint，区分旧 partial snapshot |
| 经典四条外部地震输入与配套 NTHA 时程 | 原生采样率评估代码和结果在；尚未建立公开数据文件索引 | 归档确切输入/参考文件和单位、采样率、来源记录，不能只给 CESMD 首页 |
| §4 旧 FNO 超参数图对应的完整模型索引 | 原始代码、汇总表及旧 Hugging Face 模型库已有 | 给 Fig.4–7 每个绘图系列补齐具体 config/checkpoint 路径与哈希，避免仅凭文件名猜测 |
| 原 NTHA 生成程序与参数链 | HDF5 审计能证明文件有限值与覆盖范围，不能证明历史求解器版本、收敛/失稳判据 | 将“已存数据库的模型训练复现”与“从地震记录重新生成 NTHA 数据库”分开声明；未确认版本的 Fortran 源码未冒充原生成器 |
| 明确的仓库许可证及第三方声明 | 当前仓库根目录没有 LICENSE | 作者确定代码许可，保留所依赖 neuralop 等第三方的许可声明；不能用“能下载”代替明确开源许可 |

PFA/FRS 指标的 `g` 常量在历史模块中有 9.81 与 9.80665 两种取法。本次保留各实验原取法，以匹配归档数值，未全局替换；R² 不受统一比例缩放影响，但 g 单位的误差/阈值应按具体模块解释。

## 不纳入本次公开实验配方的内容

未被当前论文报告的 descriptor GM-OOD、额外调参 scarcity/transfer、服务器排队/迁移/监控、失败探针、同步冲突文件、缓存、内部回复草稿，以及未经确认的求解器副本均未作为公开实验整理进去。原始日志中的来源路径保留作历史证据，不用作新机器的执行路径。

最终发布前应再次用最新版论文刷新映射与哈希。本次整理的结果是可审查、可验证的代码和证据包；新增 GPU 训练没有在整理阶段重跑。
