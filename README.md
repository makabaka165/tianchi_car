# 天池二手车交易价格预测

<p align="center">
  一个面向表格回归竞赛的服务器端实验工程：<br/>
  以 <strong>可复现</strong>、<strong>可回退</strong>、<strong>可持续提分</strong> 为核心约束，围绕 LightGBM + CatBoost + OOF 融合持续优化二手车价格预测表现。
</p>

<p align="center">
  <strong>Best Blend OOF MAE:</strong> <code>480.87277472593655</code>
  &nbsp;|&nbsp;
  <strong>Stable Code Commit:</strong> <code>53c36fa</code>
  &nbsp;|&nbsp;
  <strong>Current Repo Track:</strong> <code>server-first</code>
</p>

---

## 01. 项目定位

这不是一个“只跑一次 baseline”的比赛脚本仓库，而是一套针对天池二手车交易价格预测赛题构建的完整实验系统。

它解决的核心问题有三个：

- 如何把表格回归任务做成一个能持续提分的工程，而不是零散脚本集合。
- 如何在多轮调参与特征试验中，保证结果可复现、代码可回退、日志可追踪。
- 如何在 LightGBM 与 CatBoost 这两类树模型之间做合理分工，并通过 OOF 融合榨出稳定增益。

当前仓库的所有正式训练、实验记录、版本保留、提交与推送，都以服务器仓库 `~/tianchi_car` 为唯一执行环境。本地仓库只承担拉取与查看作用。

---

## 02. 当前最佳结果

当前正式保留的最佳实验为 `task8_cat_iter4600_cfg_w_eval`，核心结果如下：

| 指标 | 数值 |
|---|---:|
| LightGBM OOF MAE | `526.6174419083302` |
| CatBoost OOF MAE | `488.8264628108603` |
| Blend OOF MAE | `480.87277472593655` |
| 最优 LightGBM 权重 | `0.282` |
| 最优 CatBoost 权重 | `0.718` |
| 服务器稳定代码提交 | `53c36fa` |

结论非常明确：

- CatBoost 是当前主力得分模型。
- LightGBM 单模分数不占优，但提供了真实可用的互补信息。
- 最终高分不是来自复杂堆叠，而是来自严格验证、稳健特征和细粒度融合。

---

## 03. 为什么这个版本有效

当前高分版本不是靠某一个“神奇参数”拿到的，而是四个环节同时做对了。

### 1. 验证协议可信

- 固定 5 折交叉验证。
- 正式比较指标统一为 `MAE`。
- 面向 LightGBM 的目标统计特征采用折内生成，避免验证泄漏。
- 最终保留决策以 `user_data/metrics.json` 为准，而不是临时屏幕输出。

### 2. 特征工程强调低泄漏与稳定性

有效特征集中在以下几类：

- 日期与车龄特征
- 数值变换与比例交互特征
- 匿名变量统计聚合特征
- 组合类别特征
- 频次特征
- 缺失计数特征
- 折内目标统计特征

这套特征并不追求臃肿，而是优先保证训练集与测试集的一致性，以及模型对结构化信息的可利用性。

### 3. 模型分工明确

不是让两个模型吃同一份东西、做同一种判断，而是让它们各自承担不同角色：

- LightGBM：承担结构化补充与融合辅助视角。
- CatBoost：承担主要拟合能力和最终主力得分来源。

### 4. 融合建立在 OOF 而不是主观经验上

最终不是拍脑袋给 5:5 或 3:7 权重，而是直接在 OOF 上搜索最优线性组合。最终保留权重为：

- LightGBM `0.282`
- CatBoost `0.718`

这一步让融合从“经验修饰项”变成了正式收益来源。

---

## 04. 方法结构

### 特征工程

核心实现位于 `feature/preprocess.py`。

当前高分版本重点使用：

- `regDate` / `creatDate` 解析与车龄衍生
- `power_log1p`、`kilometer_log1p` 等数值变换
- `power_per_km`、`power_per_age` 等交互比例特征
- `v_0 ~ v_14` 的聚合统计特征
- `brand_model`、`brand_bodyType` 组合类别特征
- `brand_freq`、`model_freq`、`name_freq` 等频次特征
- `missing_count` 缺失计数特征

### LightGBM 分支

LightGBM 更偏向结构化统计建模，最终保留参数如下：

```text
n_estimators=4000
learning_rate=0.03
num_leaves=127
subsample=0.8
colsample_bytree=0.8
reg_alpha=0.2
reg_lambda=0.2
```

特征层面会接入折内目标统计信息，目标采用 `log1p(price)` 进行拟合。

### CatBoost 分支

CatBoost 是当前最强单模型，最终保留候选如下：

```text
cfg_w_depth_9_iter4600
iterations=4600
learning_rate=0.03
depth=9
l2_leaf_reg=9.0
od_wait=140
```

从实验结果看，CatBoost 在这个任务中对参数近邻变化较敏感，因此后续最值得投入算力的方向仍然是围绕它做单变量细调。

### OOF 融合

融合使用 5 折 OOF 结果做线性权重搜索，并且把权重步长细化到 `0.001`。这属于低风险、稳定有效的后处理增强。

---

## 05. 提分路径

这个仓库的提分并不是“大改一把梭”，而是沿着下面这条路线逐步推进：

1. 先搭通双模型 baseline 与提交链路。
2. 引入基础统计特征和 `log1p` 目标变换。
3. 把目标统计改成折内生成，修正验证方式。
4. 验证 LightGBM 与 CatBoost 是否应分开使用不同特征输入。
5. 确认 CatBoost 更适合回到原始 `price` 目标空间。
6. 细化 LightGBM 结构参数。
7. 细化融合权重搜索粒度。
8. 持续围绕 CatBoost 做单变量参数实验，最终把最佳 Blend OOF 压到 `480.87277472593655`。

最重要的经验不是“改了哪些参数”，而是：

- 每轮只改一个方向。
- 结果变好才保留。
- 结果变差就回退代码，但日志必须留下。

这套实验纪律，本身就是仓库价值的一部分。

---

## 06. 快速开始

### 环境

建议直接在服务器上的既有竞赛环境中运行，本仓库的正式实验环境为 `~/tianchi_car`。

```bash
cd ~/tianchi_car
```

### 全量训练与预测

```bash
python code/main.py full --experiment-note full_run
```

### 仅准备 LightGBM 缓存

```bash
python code/main.py prepare_lgb_cache --experiment-note cache_build
```

### 只跑 CatBoost 候选实验

```bash
python code/main.py catboost_only_sweep \
  --candidate cfg_w_depth_9_iter4600 \
  --experiment-note cat_iter4600_eval \
  --baseline-blend 480.87277472593655 \
  --baseline-commit 53c36fa
```

### 关键产物

- 正式结果：`user_data/metrics.json`
- 提交文件：`prediction_result/predictions.csv`
- 实验日志：`记录.md`

---

## 07. 项目结构

```text
tianchi_car/
├─ code/                  # 主入口、实验编排、融合与日志逻辑
├─ feature/               # 数据读取、清洗、特征工程
├─ model/                 # LightGBM / CatBoost 训练封装
├─ user_data/             # 缓存、metrics、中间产物
├─ prediction_result/     # 最终预测文件 predictions.csv
├─ 记录.md                # 逐轮实验日志
├─ 赛题.md                # 比赛任务与字段说明
├─ 比赛总结.md            # 高分路线与代码思路总结
└─ README.md              # 项目首页说明
```

---

## 08. 实验治理规则

这个仓库不是“分数优先、不管过程”的写法，而是明确采用服务器端可治理流程：

- 服务器是唯一正式执行环境。
- 每轮实验先记录目标和基线，再开始修改。
- 每轮实验后统一把结果追加到 `记录.md`。
- 新结果严格优于当前基线才保留代码并推送。
- 若结果不优，则代码回退到上一稳定提交，但失败日志继续保留。

这个规则的目的不是形式化，而是为了让后续每一次提分都有证据链。

---

## 09. 阅读顺序建议

如果你是第一次接触这个仓库，推荐按下面顺序理解：

1. `README.md`：先建立整体认识。
2. `赛题.md`：明确任务、字段和提交格式。
3. `比赛总结.md`：理解当前高分路线和各阶段决策。
4. `记录.md`：查看每轮实验改了什么、保留了什么、回退了什么。
5. `feature/preprocess.py`、`model/train.py`、`code/main.py`：最后回到实现层。

---

## 10. 后续优化建议

在当前版本之上继续提分，最合理的顺序依然是：

1. 继续围绕 CatBoost 做单变量近邻参数实验。
2. 连续两轮无收益后，再切换到轻量特征实验。
3. 只有单模型结果改善后，再重新搜索融合权重。
4. 不建议在同一轮里同时修改特征、参数和融合策略。

如果要继续把这个仓库往更成熟的竞赛工程推进，下一步最值得补的是：

- 更标准化的环境说明
- 更清晰的实验配置抽象
- 更明确的结果表格与模型版本索引

---

## 11. 相关文档

- [比赛总结.md](./比赛总结.md)
- [记录.md](./记录.md)
- [赛题.md](./赛题.md)

---

## 12. 总结

当前版本的核心竞争力，不是模型复杂度，而是工程质量：

- 验证方式可信。
- 特征体系稳定。
- 模型分工清楚。
- 融合策略有效。
- 实验流程可追踪、可保留、可回退。

这也是为什么它更像一个可以继续迭代的竞赛工程，而不是一份一次性提交脚本。
