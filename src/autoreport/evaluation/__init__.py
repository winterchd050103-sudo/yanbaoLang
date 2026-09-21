"""评测闭环：自建评测集 -> 跑基线 -> 指标汇总 -> 失败分析 -> 迭代优化。

这是简历上最硬的数字来源（方案 8.x）：
- 评测集从「真实入库研报」自动生成种子（gold report_id 天然可靠，零标注成本）
- 指标：检索命中率@K / 引用正确率（程序判定，完全客观）+ LLM-as-judge 回答准确率
- LangSmith 开启时同步上传评测集（tracing 由 apply_langsmith_env 全局接管），
  未开启时全部能力照常可用（本地 JSONL + Markdown 报告）
"""

from autoreport.evaluation.dataset import build_seed_dataset, load_evalset
from autoreport.evaluation.run_eval import run_eval

__all__ = ["build_seed_dataset", "load_evalset", "run_eval"]
