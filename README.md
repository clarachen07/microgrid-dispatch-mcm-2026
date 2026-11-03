# 微网购电与储能协同调度

> 2026 年全国大学生数学建模竞赛 C 题研究项目  
> *Microgrid electricity procurement and energy-storage co-dispatch for MCM 2026 Problem C.*   
> 此项目荣获2026全国大学生数学建模竞赛北京二等奖

本项目针对含光伏和储能的微网，建立不确定负载、光伏出力与实时电价下的购电和充放电模型。核心方法包括因果滚动预测、分位数风险裕度和情景滚动优化，覆盖赛题四个问题。

This repository reproduces the four primary solutions for MCM 2026 Problem C using causal rolling forecasts, quantile risk margins, and receding-horizon optimisation.

## 项目结构 / Layout

```text
data/
  official-input/        官方附件与结果模板（保留原始中文文件名）
  problem-statement.pdf  赛题 PDF
final-output/            已上交的最终论文与支撑材料，保持冻结
paper/                   原始论文源码、图表与生成表格，保持冻结
src/microgrid/           数据、预测、优化、仿真、导出与约束检查模块
tests/                   核心模型测试
run.py                   分阶段求解入口
reproduce.py             主模型完整复现入口
```

`artifacts/` 是本地生成目录，保存运行账本和导出的五个结果工作簿，不进入版本控制。

## 快速复现 / Quick start

建议使用 Python 3.9：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

完整运行问题一至问题四、导出结果并检查能量守恒、储能边界和工作簿格式：

```sh
.venv/bin/python reproduce.py
```

也可以分阶段执行：

```sh
.venv/bin/python run.py q1
.venv/bin/python run.py calibrate
.venv/bin/python run.py select
.venv/bin/python run.py main
.venv/bin/python run.py export
.venv/bin/python run.py verify --require-workbooks
```

五个结果工作簿生成于 `artifacts/exports/`。独立运行支撑代码时，可将官方附件放在 `C题/附件/`；数据读取器仍兼容这一旧路径。

## 数据与成果 / Data and results

`data/official-input/` 中保存官方附件 1–5，内容未修改。`final-output/` 中保存原始上交成果，作为冻结副本，不由当前代码重写。使用者应遵守竞赛、数据发布方及所在机构的相关规则。
