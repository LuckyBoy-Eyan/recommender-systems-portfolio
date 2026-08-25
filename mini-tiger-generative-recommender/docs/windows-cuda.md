# 在 Windows NVIDIA 显卡上训练

## 1. 需要复制的内容

把整个项目复制到 Windows，至少保留：

```text
configs/
scripts/
src/
tests/
data/kuairec_big/interactions.csv
data/kuairec_big/item_features.npy
data/kuairec_big/item_ids.npy
data/kuairec_big/stats.json
```

不需要复制 412 MB 的原始 KuaiRec ZIP。若要从已有训练检查点续跑，还应复制对应
`outputs/<实验名>/`。检查点使用 `map_location` 加载，可以在 CPU 与 CUDA 间迁移。

## 2. 创建环境

在 PowerShell 中：

```powershell
cd <项目目录>\mini-tiger-generative-recommender
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
```

随后打开 PyTorch 官方安装选择器，选择：

```text
OS: Windows
Package: Pip
Language: Python
Compute Platform: 与驱动兼容的 CUDA
```

执行页面生成的安装命令。不要先使用基础 `requirements.txt` 安装 CPU 版 torch。

## 3. 检查 CUDA 和显存

```powershell
nvidia-smi
python scripts/check_cuda.py
python -m pytest -q
```

自检会真实运行一个与主配置同尺寸的 FP16 Transformer 前向。如果这里显存不足，
先把 `configs/kuairec_big_cuda.json` 中的 `batch_size` 从 256 调为 128 或 64。

## 4. 启动全量实验

```powershell
python scripts/run_demo.py `
  --config configs/kuairec_big_cuda.json `
  --output outputs/kuairec_big_cuda
```

配置覆盖全部 7,174 用户、9,438 视频，每用户最多 100 个最近训练目标，并依次
训练 RQ-KMeans Semantic ID 和同容量 Random ID。CUDA 使用 FP16 自动混合精度，
每轮原子保存：

```text
outputs/kuairec_big_cuda/semantic_checkpoint.pt
outputs/kuairec_big_cuda/random_checkpoint.pt
```

进程中断后执行相同命令即可从下一轮继续。最终结果位于同目录的 `metrics.json`。

## 5. 关于 Faiss

Windows 配置有意使用 `rq_backend=sklearn`。9,438 个物品的 RQ-KMeans 建索引仅需
数秒，训练瓶颈在 PyTorch 模型，不值得为此在原生 Windows 编译 Faiss GPU。
如果后续目录扩大到百万级并需要 Faiss GPU，建议使用 WSL2 Ubuntu 环境。
