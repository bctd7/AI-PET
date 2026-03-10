"""
预先下载 BGE reranker 模型到本地缓存，避免正式跑检索时因网络超时失败。

用法（在项目根目录执行）：
    python download_bge_reranker.py

国内建议先设置镜像再执行（PowerShell）：
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    python download_bge_reranker.py
"""

import os
import sys

# 国内建议使用镜像，避免连接 huggingface.co 超时
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main() -> int:
    print("正在下载 BGE reranker 模型 (cross-encoder/ms-marco-MiniLM-L6-v2)，请稍候...")
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
        # 跑一次极小推理，确保权重等全部加载并缓存
        _ = model.predict([("query", "doc")])
        print("模型已下载并缓存完成。")
        return 0
    except Exception as e:
        print(f"下载或加载失败: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
