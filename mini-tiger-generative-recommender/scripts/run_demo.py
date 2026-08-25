"""MiniTIGER 端到端实验入口。

执行方式：
    python scripts/run_demo.py
    python scripts/run_demo.py --config configs/movielens.json --output outputs/movielens

main 串起完整调用链：
    读取配置
      -> load_interactions 或 make_synthetic_data
      -> build_rq_kmeans_codes（或旧索引消融）+ append_collision_token
      -> NextItemDataset
      -> SemanticIDTransformer
      -> train_model
      -> evaluate_model / evaluate_popularity
      -> 写入 metrics.json

同一流程还会使用 Random ID 再训练一次。两个模型结构和超参数相同，从而把
“编码是否包含物品语义”作为主要实验变量。
"""

from __future__ import annotations

import json
import os
import random
import sys
import argparse
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# 把项目根目录加入 import 路径，让脚本可以直接导入 src 下面的模块。
sys.path.insert(0, str(ROOT))

from src.data.dataset import NextItemDataset
from src.data.load import load_interactions
from src.data.split import temporal_leave_two_out, user_holdout_split
from src.data.synthetic import make_synthetic_data
from src.indexing.semantic_ids import (
    append_collision_token,
    build_hierarchical_codes,
    build_random_codes,
    codebook_diagnostics,
    collision_rate,
)
from src.indexing.rq_kmeans import (
    build_rq_kmeans_codes,
    load_rq_kmeans_artifact,
    save_rq_kmeans_artifact,
)
from src.models.generative import SemanticIDTransformer
from src.training.evaluate import evaluate_model, evaluate_popularity
from src.training.train import train_model


def evaluation_kwargs(config: dict) -> dict:
    """只向当前评估模式传递它支持的参数。"""
    common = {
        "batch_size": config.get("evaluation_batch_size", 32),
        "device": config.get("device", "cpu"),
        "exclude_seen": config.get("exclude_seen_items", False),
    }
    if config.get("evaluation_mode", "exact") == "exact":
        common["catalog_chunk_size"] = config.get("catalog_chunk_size", 512)
    else:
        common["beam_size"] = config.get("beam_size", 100)
    return common


def train_from_config(
    model,
    dataset,
    config: dict,
    validation_fn=None,
    checkpoint_path: Path | None = None,
):
    """把训练相关配置集中传给 train_model。"""
    return train_model(
        model,
        dataset,
        config["epochs"],
        config["batch_size"],
        config["learning_rate"],
        device=config.get("device", "cpu"),
        weight_decay=config.get("weight_decay", 0.01),
        gradient_clip_norm=config.get("gradient_clip_norm"),
        num_workers=config.get("num_workers", 0),
        validation_fn=validation_fn,
        monitor_metric=config.get("monitor_metric"),
        early_stopping_patience=config.get("early_stopping_patience"),
        checkpoint_path=checkpoint_path,
        resume=config.get("resume_training", False),
        mixed_precision=config.get("mixed_precision", False),
        amp_dtype=config.get("amp_dtype", "float16"),
    )


def make_datasets(sequences, codes, config: dict):
    """按配置建立训练、验证和测试数据集及对应原始序列。"""
    max_history = config["max_history"]
    if config.get("split_strategy") == "temporal_leave_two_out":
        train_sequences, validation_sequences, test_sequences = temporal_leave_two_out(
            sequences, config.get("min_train_sequence_length", 3)
        )
        train_data = NextItemDataset(
            train_sequences,
            codes,
            max_history,
            max_samples_per_sequence=config.get("max_train_samples_per_user"),
        )
        validation_data = NextItemDataset(
            validation_sequences, codes, max_history, last_only=True
        )
        test_data = NextItemDataset(test_sequences, codes, max_history, last_only=True)
        return (
            train_sequences,
            train_data,
            validation_data,
            test_data,
        )

    train_sequences, test_sequences = user_holdout_split(sequences)
    train_data = NextItemDataset(train_sequences, codes, max_history)
    test_data = NextItemDataset(
        test_sequences,
        codes,
        max_history,
        config.get("eval_last_only", False),
    )
    return train_sequences, train_data, None, test_data


def evaluate_splits(model, validation_data, test_data, codes, config: dict) -> dict:
    """用相同协议评估可选验证集和测试集。"""
    mode = config.get("evaluation_mode", "exact")
    kwargs = evaluation_kwargs(config)
    results = {}
    if validation_data is not None:
        results["validation"] = evaluate_model(
            model, validation_data, codes, config["topk"], mode=mode, **kwargs
        )
    results["test"] = evaluate_model(
        model, test_data, codes, config["topk"], mode=mode, **kwargs
    )
    # 旧 Demo 的 metrics.json 原本把测试指标直接放在 semantic_id 下。
    return results["test"] if validation_data is None else results


def make_validation_callback(validation_data, codes, config: dict):
    """按配置创建每轮验证回调；未启用时返回 None。"""
    if validation_data is None or not config.get("validate_every_epoch", False):
        return None
    mode = config.get("evaluation_mode", "exact")
    kwargs = evaluation_kwargs(config)

    def callback(model):
        """在当前验证集上按主配置的解码模式计算指标。"""
        return evaluate_model(
            model,
            validation_data,
            codes,
            config["topk"],
            mode=mode,
            **kwargs,
        )

    return callback


def main():
    """解析命令行配置，依次执行数据、编码、训练、评估和保存。

    参数:
        无 Python 函数参数。命令行参数由 argparse 读取：
        ``--config`` 指向相对项目根目录的 JSON 配置；
        ``--output`` 指向相对项目根目录的结果目录。

    返回:
        None。结果通过标准输出展示，并写入 ``<output>/metrics.json``。

    调用:
        文件底部的 ``if __name__ == "__main__"`` 在直接运行脚本时调用。
        它是项目顶层编排函数，调用 src 下的数据、索引、模型和训练模块。
    """
    # config 控制数据来源、模型规模、训练轮数和评估 top-k。
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/demo.json",
        help="相对项目根目录的实验配置 JSON",
    )
    parser.add_argument(
        "--output",
        default="outputs/demo",
        help="相对项目根目录的指标输出目录",
    )
    args = parser.parse_args()
    # 索引器也会在这个目录保存可复用码本，因此在训练开始前创建输出目录。
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    # JSON 配置被读成 dict；后续所有规模和超参数都从 config 取值。
    config = json.loads((ROOT / args.config).read_text())
    # 小矩阵运算在 CPU 上并非线程越多越快；主配置使用基准测得的线程数。
    if config.get("torch_num_threads") is not None:
        torch.set_num_threads(int(config["torch_num_threads"]))
    # 固定随机种子，让合成数据、KMeans、模型初始化尽量可复现。
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果配置里给了真实数据路径，就读取真实交互和物品特征。
    # 否则使用内置合成数据，便于快速跑通整个流程。
    if config.get("interactions_path") and config.get("item_features_path"):
        features, sequences = load_interactions(
            ROOT / config["interactions_path"],
            ROOT / config["item_features_path"],
            min_sequence_length=config.get("min_sequence_length", 3),
            max_sequence_length=config.get("max_sequence_length"),
        )
    else:
        features, sequences = make_synthetic_data(
            **{key: config[key] for key in [
                "num_users", "num_items", "num_topics", "feature_dim",
                "min_seq_len", "max_seq_len", "seed"
            ]}
        )
    # 工业主路径使用全局多级残差量化；旧 residual/hierarchical 仍保留为消融。
    # 未声明方法的历史配置继续走旧 residual，避免升级后静默改变已有实验。
    semantic_id_method = config.get("semantic_id_method", "residual")
    rq_result = None
    rq_manifest = None
    prebuilt_rq_dir = config.get("prebuilt_rq_index_dir")
    if prebuilt_rq_dir:
        if semantic_id_method != "rq_kmeans":
            raise ValueError("prebuilt_rq_index_dir requires semantic_id_method=rq_kmeans")
        artifact = load_rq_kmeans_artifact(ROOT / prebuilt_rq_dir)
        rq_manifest = artifact["manifest"]
        raw_codes = artifact["arrays"]["codes"]
        if len(raw_codes) != len(features):
            raise ValueError(
                "prebuilt RQ catalog differs from loaded item feature catalog"
            )
        if rq_manifest["codebook_sizes"] != config["codebook_sizes"]:
            raise ValueError("prebuilt RQ codebook sizes differ from config")
    elif semantic_id_method == "rq_kmeans":
        rq_result = build_rq_kmeans_codes(
            features,
            config["codebook_sizes"],
            seed,
            backend=config.get("rq_backend", "auto"),
            pca_dim=config.get("rq_pca_dim"),
            whiten=config.get("rq_whiten", True),
            l2_normalize=config.get("rq_l2_normalize", True),
            niter=config.get("rq_niter", 25),
            nredo=config.get("rq_nredo", 3),
            use_gpu=config.get("rq_use_gpu", False),
            max_balance_ratio=config.get("rq_max_balance_ratio", 1.25),
            resolve_collisions=config.get("rq_resolve_collisions", True),
            minibatch_threshold=config.get("minibatch_kmeans_threshold", 5000),
        )
        raw_codes = rq_result.codes
        rq_manifest = rq_result.manifest()
        save_rq_kmeans_artifact(rq_result, output)
    else:
        raw_codes = build_hierarchical_codes(
            features,
            config["codebook_sizes"],
            seed,
            config.get("minibatch_kmeans_threshold", 5000),
            semantic_id_method,
        )
    # 追加 tail token，保证每个物品最终有唯一完整 code。
    codes, tail_size = append_collision_token(raw_codes)
    # 模型每个输出头都需要知道对应层的词表大小；最后补上 tail 层大小。
    semantic_sizes = [*config["codebook_sizes"], tail_size]
    # KuaiRec 使用逐用户时间切分；旧 Demo 继续支持按用户切分以复现实验。
    train_sequences, train_data, validation_data, test_data = make_datasets(
        sequences, codes, config
    )
    # 初始化生成式推荐模型：Transformer 编码历史，GRUCell 自回归生成 code。
    torch.manual_seed(seed)
    model = SemanticIDTransformer(
        semantic_sizes, config["max_history"], config["hidden_dim"],
        config["num_heads"], config["num_layers"], config.get("feedforward_dim")
    )
    # 训练 Semantic ID 模型，并在测试集上评估。
    train_from_config(
        model,
        train_data,
        config,
        make_validation_callback(validation_data, codes, config),
        output / "semantic_checkpoint.pt",
    )
    semantic_metrics = evaluate_splits(
        model, validation_data, test_data, codes, config
    )

    random_metrics = None
    if config.get("run_random_baseline", True):
        # Random ID 使用同样容量和切分，是检验“语义编码本身”的关键对照。
        random_codes, random_sizes = build_random_codes(
            len(features), config["codebook_sizes"], seed
        )
        _, random_train, random_validation, random_test = make_datasets(
            sequences, random_codes, config
        )
        # 重置初始化与 DataLoader 随机流，使两组模型的比较更可控。
        torch.manual_seed(seed)
        random_model = SemanticIDTransformer(
            random_sizes,
            config["max_history"],
            config["hidden_dim"],
            config["num_heads"],
            config["num_layers"],
            config.get("feedforward_dim"),
        )
        train_from_config(
            random_model,
            random_train,
            config,
            make_validation_callback(random_validation, random_codes, config),
            output / "random_checkpoint.pt",
        )
        random_metrics = evaluate_splits(
            random_model, random_validation, random_test, random_codes, config
        )
    # 汇总数据规模、Semantic ID 结果、Random ID 结果、热门基线和碰撞率。
    metrics = {
        "experiment": {
            "split_strategy": config.get("split_strategy", "user_holdout"),
            "evaluation_mode": config.get("evaluation_mode", "exact"),
            "device": config.get("device", "cpu"),
            "torch_num_threads": torch.get_num_threads(),
            "mixed_precision": config.get("mixed_precision", False),
            "amp_dtype": config.get("amp_dtype", "float16"),
            "semantic_id_method": semantic_id_method,
            "prebuilt_rq_index_dir": prebuilt_rq_dir,
        },
        "data": {
            "users": len(sequences),
            "items": len(features),
            "train_samples": len(train_data),
            "validation_samples": len(validation_data) if validation_data else 0,
            "test_samples": len(test_data),
        },
        "semantic_id": semantic_metrics,
        "semantic_id_training": model.training_history,
        "random_id": random_metrics,
        "popularity": {
            "validation": evaluate_popularity(
                validation_data,
                train_sequences,
                config["topk"],
                exclude_seen=config.get("exclude_seen_items", False),
            )
            if validation_data
            else None,
            "test": evaluate_popularity(
                test_data,
                train_sequences,
                config["topk"],
                exclude_seen=config.get("exclude_seen_items", False),
            ),
        }
        if validation_data
        else evaluate_popularity(
            test_data,
            train_sequences,
            config["topk"],
            exclude_seen=config.get("exclude_seen_items", False),
        ),
        "raw_codebook": codebook_diagnostics(raw_codes, config["codebook_sizes"]),
        "rq_kmeans": rq_manifest,
        "raw_semantic_id_collision_rate": collision_rate(raw_codes),
        "resolved_semantic_id_collision_rate": collision_rate(codes),
    }
    # 保存 metrics.json，方便 README 或实验报告引用。
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(output / "semantic_codes.npy", codes)
    if config.get("save_model", False):
        torch.save(model.state_dict(), output / "semantic_model.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
