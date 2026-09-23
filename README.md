# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 运行测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

GA 与 PSO 的共同测试覆盖：首个候选失败、搜索中途失败、合法惩罚（几何违规与 `InfeasibleCandidate` 声明）、全部候选无效，以及创建期配置校验。

## 优化器故障处理约定

GA/PSO 明确区分两类情况：

- **候选不可行**：几何约束违规，或目标函数抛出 `wind_farm_opt.optimization.evaluation.InfeasibleCandidate` 主动声明不可行。此类候选按 `-penalty_factor` 的明确惩罚计入适应度，搜索继续。
- **评估器故障**：目标函数返回 NaN/Inf 等非有限值，或抛出 `InfeasibleCandidate` 以外的异常（如维度不匹配）。此时立即抛出 `ObjectiveFailureError` 终止搜索，异常中保留故障候选的索引、布局 `(N, 2)`、代/迭代编号和原始异常；CLI 以退出码 2 结束，`results.json` 写入 `optimization_status.success=false`，不会把失败运行伪装成有效最优解。

当整个运行没有任何可行候选时，`OptimizeResult.best_feasible` 为 `False`，`best_*` 仅为惩罚最小的诊断布局，不构成有效最优解。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。
