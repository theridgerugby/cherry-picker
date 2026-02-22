# config.py — 所有参数在这里改，不需要动其他文件

DOMAIN = "Sparse Representation"

# arXiv 搜索关键词（可以加多个，用 AND/OR 连接）
ARXIV_QUERY = "ti:sparse AND ti:representation OR ti:\"dictionary learning\" OR ti:\"sparse coding\" OR ti:\"compressed sensing\""

# 拉取最近多少天的论文
DAYS_BACK = 30

# 最多拉取多少篇
MAX_PAPERS = 10

# 每次 Agent 至少处理多少篇才能结束
MIN_PAPERS_TO_PROCESS = 5

# Gemini 模型选择
# ── 双模型策略 ──────────────────────────────────────────────────────────────
# Fast model：用于报告主体生成 / 方法论矩阵 / 技能提取（高吞吐、低延迟）
GEMINI_MODEL_FAST = "gemini-2.5-flash"
# Deep model：用于跨论文推理（趋势分析、间隙推断）— 允许 thinking
GEMINI_MODEL = "gemini-2.5-pro"
# Thinking token 预算（仅对 thinking model 有效）
# 越低 = 越快；建议平衡点 1024–4096；设为 0 = 禁用 thinking
THINKING_BUDGET = 2048

# 向量数据库存储路径（本地）
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION = "arxiv_papers"

# 报告输出路径
REPORT_OUTPUT_PATH = "./report.md"

# 技能提取缓存路径（避免重复调用 LLM）
SKILL_CACHE_PATH = "./skill_cache.json"

# ── 行业趋势分析配置 ──────────────────────────────────────────────────────────
CREDIBILITY_THRESHOLD = 30   # 参与趋势分析的最低可信度分数
TREND_TOP_N = 15             # 趋势分析取前 N 篇高分论文
EXTENDED_DAYS_BACK = 180     # 趋势分析回溯天数（6 个月）

# ── 置信度门控阈值 ─────────────────────────────────────────────────────────────
MIN_PAPERS_FOR_TREND_ANALYSIS = 8    # 运行宏观趋势分析所需的最少高可信度论文数
MIN_PAPERS_FOR_ROADMAP = 10          # 运行学习路线图生成所需的最少论文数
MIN_PAPERS_FOR_COMPARISON = 3        # 运行方法论对比矩阵所需的最少论文数


# Low-confidence report mode:
# - "rich": generate full report (themes/gaps/reading order), but skip trend+roadmap
# - "cards_only": output single-paper cards only
LOW_CONFIDENCE_REPORT_MODE = "rich"

# ── 空内容识别短语（用于智能章节隐藏）────────────────────────────────────────
EMPTY_SECTION_PHRASES = [
    "not explicitly mentioned",
    "do not explicitly",
    "no unsolved problems",
    "each paper addresses a specific problem",
    "not mentioned",
    "no information available",
    "none identified",
    "no gaps identified",
    "n/a",
]

# ── arXiv 类别参考（供文档使用，不参与代码逻辑）─────────────────────────────
ARXIV_CATEGORY_MAP = {
    "cs.LG":   "Machine Learning",
    "cs.AI":   "Artificial Intelligence",
    "cs.CV":   "Computer Vision",
    "cs.CL":   "NLP",
    "cs.NE":   "Neural Computing",
    "eess.SP": "Signal Processing",
    "stat.ML": "Statistics ML",
    "math.OC": "Optimization",
}
