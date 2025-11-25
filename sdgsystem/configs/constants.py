# file extensions
DEFAULT_EXPORT_FORMAT = "jsonl"


# settings for keyword extractor
DEFAULT_KEYWORDS_EXTRACT_EXAMPLES = [
    {"你的任务是以选择题形式回答 CFA 考试问题。你需要从三个选项（例如 A、B、C）中选择正确答案。题目涵盖资产估值、应用投资工具和概念分析各种投资、投资组合管理、财富规划、伦理与专业标准等。": "['Chinese Finance', 'CFA', 'Chinese CFA', 'Asset Valuation', 'Investment Tools', 'Portfolio Management', 'Wealth Planning', 'Ethics', 'Professional Standards']"},
    {"The task evaluates a model’s ability to answer multiple-choice questions from the United States Medical Licensing Examination (USMLE). These questions test professional-level knowledge across a broad range of medical domains, including physiology, pathology, pharmacology, and clinical reasoning. The task requires models to understand complex biomedical context, reason across multiple pieces of information, and choose the correct answer from 4 options.": "['USMLE', 'Medical Licensing', 'Physiology', 'Pathology', 'Pharmacology', 'Clinical Reasoning', 'Biomedical Context']"},
    {"You are given a word problem involving basic arithmetic, algebra, or geometry. Your task is to carefully read the problem and provide a step-by-step solution for it": "['Mathematics', 'Arithmetic', 'Algebra', 'Geometry', 'math reasoning']"},
    {"你的任务是回答具有挑战性的研究生水平的选择题，内容涵盖物理、化学和生物学，需要深厚的学科知识、复杂的推理、计算和信息综合能力。": "['Chinese Graduate Exam', 'Chinese Exam', 'Chinese Reasoning', 'Physics', 'Chemistry', 'Biology', 'Advanced Reasoning']"}
]

# settings for API LLM
DEFAULT_API_PROVIDER = "openai"
DEFAULT_TEMPERATURE = 1.0
# message roles
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# settings for answer extraction
DEFAULT_ANSWER_INSTRUCTION = "Output your final answer after <answer>"

# retry settings for LLMs
DEFAULT_MAX_RETRY_ATTEMPTS = 10
DEFAULT_RETRY_BASE_DELAY = 2


# settings for local LLM
DEFAULT_LOCAL_MODEL_LEN = 8192
DEFAULT_GPU_UTILIZATION = 0.9


# settings for task
DEFAULT_TASK_NAME = "default_task"
# settings for local task
# retrieval
DEFAULT_RETRIEVAL_METHOD = "bm25"
DEFAULT_RETRIEVAL_TOP_K = 1000

# parsing
DEFAULT_PARSING_METHOD = "mineru"

# settings for majority voting
DEFAULT_N_VOTING = 16
DEFAULT_VOTING_METHOD = "exact_match"

# settings for evaluation
DEFAULT_EVAL_BATCH_SIZE = 100

# settings for answer comparison
DEFAULT_COMPARISON_METHOD = "exact_match"

# settings for rewrite
DEFAULT_REWRITE_METHOD = "difficulty_adjust"
DEFAULT_EASIER_TEMPERATURE = 0.9
DEFAULT_HARDER_TEMPERATURE = 1.1


# # settings for web task
DEFAULT_WEB_DATASET_LIMIT = 5
DEFAULT_WEB_SAMPLE_LIMIT = 5
