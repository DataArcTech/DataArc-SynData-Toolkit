# file extensions
DEFAULT_EXPORT_FORMAT = "jsonl"

# settings for keyword extractor
DEFAULT_KEYWORDS_EXTRACT_EXAMPLES = [
    {"Your task is to answer CFA exam questions in multiple-choice format. You need to select the correct answer from three options (e.g., A, B, C). Topics include asset valuation, investment tools and concepts, portfolio management, wealth planning, ethics, and professional standards.": "['Finance', 'CFA', 'Asset Valuation', 'Investment Tools', 'Portfolio Management', 'Wealth Planning', 'Ethics', 'Professional Standards']"},
    {"The task evaluates a model's ability to answer multiple-choice questions from the United States Medical Licensing Examination (USMLE). These questions test professional-level knowledge across a broad range of medical domains, including physiology, pathology, pharmacology, and clinical reasoning. The task requires models to understand complex biomedical context, reason across multiple pieces of information, and choose the correct answer from 4 options.": "['USMLE', 'Medical Licensing', 'Physiology', 'Pathology', 'Pharmacology', 'Clinical Reasoning', 'Biomedical Context']"},
    {"You are given a word problem involving basic arithmetic, algebra, or geometry. Your task is to carefully read the problem and provide a step-by-step solution for it.": "['Mathematics', 'Arithmetic', 'Algebra', 'Geometry', 'Math Reasoning']"},
    {"Your task is to answer challenging graduate-level multiple-choice questions covering physics, chemistry, and biology. This requires deep subject knowledge, complex reasoning, calculations, and information synthesis.": "['Graduate Exam', 'Physics', 'Chemistry', 'Biology', 'Advanced Reasoning', 'Science']"}
]

# settings for HuggingFace keyword extractor (broader keywords for dataset search)
DEFAULT_HF_KEYWORDS_EXTRACT_EXAMPLES = [
    {"Your task is to answer CFA exam questions in multiple-choice format covering asset valuation, portfolio management, and ethics.": "['finance', 'qa', 'exam']"},
    {"The task evaluates a model's ability to answer USMLE questions testing knowledge across medical domains including physiology, pathology, and pharmacology.": "['medical', 'qa', 'exam']"},
    {"You are given a word problem involving basic arithmetic, algebra, or geometry. Provide a step-by-step solution.": "['math', 'gsm8k', 'reasoning']"},
    {"Answer questions about images including charts, diagrams, and natural scenes.": "['vqa', 'image', 'qa']"},
    {"Your task is to answer challenging graduate-level questions covering physics, chemistry, and biology.": "['science', 'qa', 'exam']"}
]

# settings for API LLM
DEFAULT_API_PROVIDER = "openai"
DEFAULT_TEMPERATURE = 1.0

# message roles
ROLE_USER = "user"

# settings for answer extraction
DEFAULT_ANSWER_TAG = "<answer>"
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

# settings for answer comparison
DEFAULT_COMPARISON_METHOD = "exact_match"

# settings for rewrite
DEFAULT_REWRITE_METHOD = "difficulty_adjust"
DEFAULT_EASIER_TEMPERATURE = 0.9
DEFAULT_HARDER_TEMPERATURE = 1.1

# settings for web task
DEFAULT_WEB_DATASET_LIMIT = 5
DEFAULT_WEB_SAMPLE_LIMIT = 5
