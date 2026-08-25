"""pytest 全局启动配置。

pytest 会在收集测试时自动导入 conftest.py，不需要其他文件显式调用。
这里限制底层并行线程，仅影响运行资源与稳定性，不改变模型计算逻辑。
"""

import os


# 测试启动前限制底层数学库线程数，避免 PyTorch、sklearn 同时抢线程导致运行不稳定。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
