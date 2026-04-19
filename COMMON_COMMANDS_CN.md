# Neural-CDE-based-IceTransfer 常用命令说明

本文档整理了这个项目在 Windows PowerShell 下最常用的命令，并优先给出当前仓库状态下可直接运行的写法。

## 1. 进入项目目录

```powershell
Set-Location S:\STU-Papers\My_Papers\Neural-CDE-based-IceTransfer
```

## 2. 启动 `EXP2-B-tc2020.yaml` 实验

### 2.1 当前仓库状态下可直接运行的命令

说明：我已经在当前环境里验证过，系统 `python` 还不能直接导入 `lakeice_ncde`，因此需要先把 `src` 加到 `PYTHONPATH`。

```powershell
Set-Location S:\STU-Papers\My_Papers\Neural-CDE-based-IceTransfer
$env:PYTHONPATH = "src"
python -m lakeice_ncde run --config configs/experiments/EXP2-B-tc2020.yaml
```

如果你想直接使用你给出的绝对路径，也可以这样写：

```powershell
Set-Location S:\STU-Papers\My_Papers\Neural-CDE-based-IceTransfer
$env:PYTHONPATH = "src"
python -m lakeice_ncde run --config S:\STU-Papers\My_Papers\Neural-CDE-based-IceTransfer\configs\experiments\EXP2-B-tc2020.yaml
```

### 2.2 完成可编辑安装后的标准命令

先安装依赖并做可编辑安装：

```powershell
Set-Location S:\STU-Papers\My_Papers\Neural-CDE-based-IceTransfer
python -m pip install -r requirements.txt
python -m pip install -e .
```

安装后可直接运行：

```powershell
python -m lakeice_ncde run --config configs/experiments/EXP2-B-tc2020.yaml
```

或者使用脚本命令：

```powershell
lakeice-ncde run --config configs/experiments/EXP2-B-tc2020.yaml
```

## 3. 查看 CLI 帮助

当前未安装时：

```powershell
$env:PYTHONPATH = "src"
python -m lakeice_ncde --help
```

安装后：

```powershell
python -m lakeice_ncde --help
```

## 4. 运行单个实验

### EXP0 预训练

```powershell
$env:PYTHONPATH = "src"
python -m lakeice_ncde run --config configs/experiments/EXP0_pretrain_autoreg.yaml
```

### EXP1 迁移实验

```powershell
$env:PYTHONPATH = "src"
python -m lakeice_ncde run --config configs/experiments/EXP1_transfer_autoreg.yaml
```

### EXP2 Stefan 物理约束实验

```powershell
$env:PYTHONPATH = "src"
python -m lakeice_ncde run --config configs/experiments/EXP2_transfer_autoreg_stefan.yaml
```

### EXP2-B TC2020 曲线物理约束实验

```powershell
$env:PYTHONPATH = "src"
python -m lakeice_ncde run --config configs/experiments/EXP2-B-tc2020.yaml
```

## 5. 运行批量实验

### 一次运行当前三组主实验

```powershell
$env:PYTHONPATH = "src"
python -m lakeice_ncde run --config configs/experiments/Run-ALL.yaml
```

`Run-ALL.yaml` 当前会并行运行：

- `EXP0_pretrain_autoreg.yaml`
- `EXP1_transfer_autoreg.yaml`
- `EXP2_transfer_autoreg_stefan.yaml`

### 只跑 EXP2 批量入口

```powershell
$env:PYTHONPATH = "src"
python -m lakeice_ncde run --config configs/experiments/Run-EXP2.yaml
```

## 6. 参数搜索

### README 里提到的参数搜索配置

```powershell
$env:PYTHONPATH = "src"
python -m lakeice_ncde search --config configs/search/参数搜索.yaml
```

### 其他搜索配置

```powershell
Get-ChildItem configs\search
```

例如：

```powershell
$env:PYTHONPATH = "src"
python -m lakeice_ncde search --config configs/search/EXP2_parallel_safe_search.yaml
```

## 7. 使用覆盖参数

### 7.1 叠加额外 YAML 覆盖文件

```powershell
$env:PYTHONPATH = "src"
python -m lakeice_ncde run --config configs/experiments/EXP2-B-tc2020.yaml --override your_override.yaml
```

### 7.2 直接在命令行里改配置值

```powershell
$env:PYTHONPATH = "src"
python -m lakeice_ncde run --config configs/experiments/EXP2-B-tc2020.yaml --set train.batch_size=64 --set train.learning_rate=0.0005
```

`--set` 支持点路径写法，格式是 `键路径=值`。

## 8. 测试命令

### 运行全部测试

```powershell
python -m pytest -q
```

### 只验证实验配置继承

```powershell
python -m pytest tests/test_experiment_config_inheritance.py -q
```

我已经在当前仓库里执行过上面这条配置继承测试，结果是 `5 passed`。

## 9. 输出位置

### 单实验输出

通常在：

```text
outputs/runs/<experiment_name>/
```

### 批量实验输出

通常在：

```text
outputs/runs/Run-ALL/
```

### 参数搜索输出

输出目录由搜索配置中的 `search.output_root` 决定。

## 10. 常见问题

### 出现 `No module named lakeice_ncde`

说明当前 Python 环境还没有安装这个包。解决方法有两种：

```powershell
$env:PYTHONPATH = "src"
```

或者执行：

```powershell
python -m pip install -e .
```

### 想恢复默认环境变量

```powershell
Remove-Item Env:PYTHONPATH
```
