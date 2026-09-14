# 微网购电与储能协同调度

> 2026 年全国大学生数学建模竞赛 C 题研究项目  
> *Microgrid electricity procurement and energy-storage co-dispatch for MCM 2026 Problem C.*

本项目研究含光伏与储能的微网如何在不确定负载、光伏出力和实时电价下制定购电与充放电策略。方案使用因果滚动预测、分位数风险裕度和情景滚动优化，覆盖赛题的四个问题，并提供可复现的代码、论文源码、官方输入和最终交付物。

This repository is a reproducible mathematical-modelling project for the 2026 MCM Problem C. It combines causal rolling forecasts, quantile risk margins, and scenario-based receding-horizon optimisation for microgrid procurement and battery dispatch.

## 方法概览 / Approach

- 统一 10 分钟功率与电量时间轴，校验能量守恒、储能边界和结果工作簿结构。
- 使用历史同星期、近期曲线与官方先验构造因果预测；问题二使用残差分位数形成风险裕度。
- 对问题三和问题四实施滚动更新、价格情景与购电调整结算，并通过独立核验与对照实验评估。

## 项目结构 / Layout

```text
data/
  official-input/        官方题目附件与结果模板（保留原始中文文件名）
  problem-statement.pdf  赛题 PDF
final-output/            最终交付：paper.pdf、supporting-materials.zip
paper/                   LaTeX 论文源码、图表和生成表格
src/microgrid/           数据、预测、优化、仿真、导出与核验模块
tests/                   自动化测试
docs/methodology.md      建模口径与复现说明
run.py                   分阶段计算入口
reproduce.py             全流程复现入口
package.py               最终交付打包与只读校验入口
```

`artifacts/` 是本地生成目录，不进入版本控制。它保存运行账本、校准结果、导出的工作簿和验证报告，可随时重新生成。

## 快速复现 / Quick start

项目建议使用 Python 3.9。创建虚拟环境并安装固定依赖：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

按顺序运行：

```sh
.venv/bin/python run.py q1
.venv/bin/python run.py calibrate
.venv/bin/python run.py select
.venv/bin/python run.py main
.venv/bin/python run.py export --results-only
.venv/bin/python run.py verify --require-workbooks
```

完整重现实验、论文表格和交付包使用：

```sh
.venv/bin/python reproduce.py
```

运行 `export` 生成的五个结果工作簿位于 `artifacts/exports/`；正式提交的只读副本位于 `final-output/`。

For a standalone copy of the supporting-materials ZIP, the official files are intentionally excluded. Put them at `C题/附件/` beside `run.py`; the loader retains this compatibility fallback.

## 最终交付 / Final deliverables

- `final-output/paper.pdf`：86 页匿名论文。
- `final-output/supporting-materials.zip`：可运行源码、依赖说明、五个结果工作簿与 AI 工具使用说明；不包含原始题目数据、缓存或 macOS 隐藏文件。

执行只读交付校验：

```sh
.venv/bin/python package.py --check-only
```

## 数据与使用说明 / Data and use

`data/official-input/` 保留竞赛发布的附件 1–5，文件内容未作修改，仅调整其所在目录以便项目结构清晰。结果仅用于本项目的数学建模复现与研究展示；使用者应遵守竞赛、数据发布方及所在机构的规则。最终论文和支撑材料中的 AI 使用披露以实际参赛队核验内容为准。
