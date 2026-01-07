"""
Centralized prompts for the entire Dataarc_SDG project.

All LLM prompts used across retrieval, generation, evaluation, etc.
"""

# ============================================================================
# PART1: LOCAL TASK PROMPTS
# ============================================================================

# ============================================================================
# RETRIEVAL MODULE PROMPTS
# ============================================================================

DEFAULT_KEYWORD_EXTRACTION_PROMPT = (
    "You can summarize the domain of this task: {task_instruction} "
    "into a list of relevant keywords. You can refer to these task examples {demo_examples}. "
    "Output only a Python list of keywords like [\"keyword1\", \"keyword2\", \"keyword3\"]."
)

HF_KEYWORD_EXTRACTION_PROMPT = (
    "Extract search keywords for finding datasets on HuggingFace related to this task: {task_instruction}\n\n"
    "Generate BROAD, GENERIC keywords that are likely to match dataset names on HuggingFace. "
    "Use simple, common terms rather than specific technical jargon.\n\n"
    "Good keywords: 'chart', 'math', 'medical', 'science', 'qa', 'vqa', 'reasoning'\n"
    "Bad keywords: 'multi-step arithmetic', 'USMLE pathology', 'graduate-level physics'\n\n"
    "Examples: {demo_examples}\n\n"
    "Output only a Python list of keywords like [\"keyword1\", \"keyword2\", \"keyword3\"]."
)


# ============================================================================
# GENERATION MODULE PROMPTS
# ============================================================================

TEXT_META_PROMPT = """
As a DatasetGenerator, your task is to generate one new example (`input` and `output`) based on the [new instruction], [reference passage], and [few-shot examples]. Please provide a JSON dictionary response that includes the new `input` and its corresponding `output`. Use the `input` and `output` keys in the dictionary.
Try you best to ensure that the input and output you generate are distinct from the provided examples while maintaining a diverse, detailed, precise, comprehensive, and high-quality response.
CRITICAL: You MUST strictly follow the output format instruction, especially any answer formatting requirements (e.g., answer markers, tags, or special formatting). The generated output must exactly match the specified format.
"""

PATTERN_GENERATION_PROMPT = """Given you the provided task instruction, input/output format instructions, and demonstration examples:
Task Instruction: {task_instruction}
Input Format Instruction: {input_instruction}
Output Format Instruction: {output_instruction}
Demonstration Examples: {demo_examples}
Our sample consists of the input and output part.
Your output should be a general summary of the task sample pattern.
In your pattern, new and undiscovered general sample format information are preferred.
Don't mention any specific and particular sample in your summary."""

HARDER_SAMPLE_PROMPT = """The current sample is overly simplistic and can be solved effortlessly by the model. Please generate an alternative and task-similar sample that presents a significantly more challenging
and intricate problem—one that requires multi-step reasoning, creative problem-solving, and deeper analytical thought.

Current sample: {sample}

You MUST follow these format instructions when generating the harder sample:
Input Format: {input_instruction}
Output Format: {output_instruction}

Only output the revised sample in the python dictionary form with 'input' and 'output' keys."""

SIMPLER_SAMPLE_PROMPT = """The current sample is too hard and can not be solved by the model. Please generate an alternative and task-similar sample that presents a simpler sample or a sub-problem of the original sample.

Current sample: {sample}

You MUST follow these format instructions when generating the simpler sample:
Input Format: {input_instruction}
Output Format: {output_instruction}

Only output the revised sample in the python dictionary form with 'input' and 'output' keys."""

# Image-aware rewrite prompts (for VLM)
HARDER_SAMPLE_WITH_IMAGE_PROMPT = """Look at the provided image. The current question-answer pair about this image is overly simplistic and can be solved effortlessly by the model. Please generate an alternative question-answer pair about the SAME image that presents a significantly more challenging and intricate problem—one that requires multi-step reasoning, creative problem-solving, and deeper analytical thought.

Current sample: {sample}

You MUST follow these format instructions when generating the harder sample:
Input Format: {input_instruction}
Output Format: {output_instruction}

Only output the revised sample in the python dictionary form with 'input' and 'output' keys. The question must still be answerable by looking at the provided image."""

SIMPLER_SAMPLE_WITH_IMAGE_PROMPT = """Look at the provided image. The current question-answer pair about this image is too hard and can not be solved by the model. Please generate an alternative question-answer pair about the SAME image that presents a simpler question or a sub-problem of the original question.

Current sample: {sample}

You MUST follow these format instructions when generating the simpler sample:
Input Format: {input_instruction}
Output Format: {output_instruction}

Only output the revised sample in the python dictionary form with 'input' and 'output' keys. The question must still be answerable by looking at the provided image."""

LLM_JUDGE_VOTING_PROMPT = """You are an expert judge tasked with selecting the best answer from multiple candidates.

Question: {question}

Candidate Answers:
{answers}

Instructions:
1. Evaluate each answer for correctness, completeness, and clarity
2. Identify which answers are semantically equivalent or express the same core idea
3. Select the answer index that represents the best/most common correct answer
4. If multiple answers are equivalent, choose the one that is most clear and concise

Output ONLY the index number (0-{max_index}) of the best answer. Do not include any explanation or additional text.

Best answer index:"""


# ============================================================================
# EVALUATION MODULE PROMPTS
# ============================================================================

LLM_JUDGE_COMPARISON_PROMPT = """You are an expert evaluator tasked with comparing a predicted answer against a ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted}

Instructions:
1. Determine if the predicted answer is semantically equivalent to the ground truth
2. For numeric answers, check if the values are equal (within reasonable precision)
3. For textual answers, check if they convey the same meaning
4. Minor formatting differences, synonyms, or paraphrasing should be considered correct
5. The answer must be factually correct to be considered a match

Output ONLY one word: "correct" if the predicted answer matches the ground truth, or "incorrect" if it does not.

Evaluation:"""


# ============================================================================
# PART2: HUGGINGFACE TASK PROMPTS
# ============================================================================

FIELD_FILTER_PROMPT = """You are a data field identifier that determines which fields in a JSON-like object represent the instruction's input and output.

Given a JSON structure or similar text, your task is to analyze its keys and contents to decide:
- which field represents the **input** (the user's question, instruction, or request)
- which field represents the **output** (the answer, response, or completion)

### Output format:
Return a JSON object with two keys:
{{
  "input": "<name of the input field>",
  "output": "<name of the output field>"
}}

### Rules:
1. Choose the field names that most likely correspond to the instruction (input) and answer (output).
2. Only select from the provided legal keys.
3. If you cannot identify clear input and output fields, return:
   {{
     "input": null,
     "output": null
   }}
4. Do NOT infer or fabricate field names not present in the text.
5. Only output the field names, not their contents.

### Example
Input: "{{'question': 'If an angle measures 120 degrees, what is its reference angle?', 'answer': 'The reference angle is found by subtracting ...', 'topic': 'Trigonometry Basics'}}"
Legal Keys: "['question', 'answer', 'topic']"

Output:
{{
  "input": "question",
  "output": "answer"
}}

Input: {row}
Legal Keys: {legal_keys}

Output:
"""

FORMAT_CONVERSION_PROMPT = """You are a format conversion assistant that rewrites given input-output pairs from one format to another.

### Task
You are given:
- **Original Input:** a user's instruction or question.
- **Original Output:** the model's answer or completion.
- **Input Format:** description of how the input is currently formatted.
- **Output Format:** description of how it should be formatted after conversion.

Your task:
- Rewrite BOTH the input and output according to the new format.
- Preserve the original content and meaning of the input and output.
- Follow the target format conventions strictly (tone, structure, delimiters, etc.).
- Do not add explanations, only provide the final formatted content.

### Output format
Return only a JSON object:
{{
  "input": "<rewritten input>",
  "output": "<rewritten output>"
}}

---

Original Input:
{input}

Original Output:
{output}

Input Format:
{input_format}

Output Format:
{output_format}

Return:
"""

INSTRUCTION_JUDGE_PROMPT = """You are an expert LLM evaluator for instruction-tuning datasets. Your goal is to assess how helpful and appropriate an instruction sample is for training a model on a specific task.
You will be given:
1. A **Task Definition** – describing the target task the model should learn.
2. An **Instruction Sample** – containing an example instruction (and optionally a response).

You should evaluate the instruction sample across **five criteria**, using the definitions below:

---

### Evaluation Criteria

1. **Relevance (0–10)**  
   How well does the instruction sample align with the task definition and objectives?  
   - 0–2: Completely unrelated or off-task  
   - 3–5: Somewhat relevant but not fully aligned  
   - 6–8: Mostly relevant and supportive of the task goal  
   - 9–10: Perfectly aligned and directly useful for the target task

2. **Correctness (0–10)**  
   Is the response factually accurate and logically valid given the instruction?  
   - 0–2: Incorrect or nonsensical  
   - 3–5: Partially correct or includes minor errors  
   - 6–8: Mostly correct and sound  
   - 9–10: Fully correct, precise, and logically consistent  
   *(If no response is provided, evaluate correctness based on the expected answer type and structure.)*

3. **Helpfulness (0–10)**  
   Does the response provide complete, informative, and useful content that effectively fulfills the instruction?  
   - 0–2: Useless or irrelevant  
   - 3–5: Somewhat useful but incomplete or generic  
   - 6–8: Helpful and reasonably detailed  
   - 9–10: Extremely helpful, comprehensive, and valuable for learning

4. **Clarity (0–10)**  
   Is the instruction easy to understand, unambiguous, and grammatically clear?  
   - 0–2: Confusing, vague, or poorly written  
   - 3–5: Understandable but could be clearer  
   - 6–8: Clear and well-structured  
   - 9–10: Perfectly clear, concise, and unambiguous

5. **Difficulty (0–10)**  
   How challenging is this instruction–response pair relative to the target task?  
   - 0–2: Too trivial or overly complex for the task  
   - 3–5: Basic difficulty, may not stimulate learning  
   - 6–8: Appropriately challenging  
   - 9–10: Ideal level of difficulty to promote robust learning and reasoning

---

### Output Format

Return your evaluation in strict **JSON** format as follows:

{{
  "Relevance": "<0–10>",
  "Correctness": "<0–10>",
  "Helpfulness": "<0–10>",
  "Clarity": "<0–10>",
  "Difficulty": "<0–10>"
}}

---

### Inputs

**Task Definition:**  
{task_description}

**Instruction Sample:**  
{instruction_sample}

---

### Output
"""

SOLVABLE_JUDGE_PROMPT = """Given the instruction sample below and the model's attempted solution, determine if the model successfully solved the problem.
Instruction Sample: {instruction_sample}
Model's Solution: {solution}
Return "True" if the model solved it correctly, otherwise return "False".
"""


# ============================================================================
# PART3: DISTILLATION TASK PROMPTS
# ============================================================================

# SDG Distillation prompt for batch generation (generates multiple samples at once)
SDG_DISTILL_BATCH_GENERATION_PROMPT = """You are a DatasetGenerator. Your task is to generate high-quality training examples based on the provided instructions and optional demonstration examples.

## Task Instruction
{task_instruction}

{input_instruction_section}
{output_instruction_section}
{pattern_section}
{demo_examples_section}

## Your Task
Generate {batch_size} DIVERSE training examples as a JSON array.
Each example should be a dictionary with 'input' and 'output' keys.

IMPORTANT Requirements:
- Each example must be DISTINCT and cover DIFFERENT aspects of the task
- Ensure high diversity in topics, difficulty levels, and approaches
- Examples should be detailed, comprehensive, and high-quality
- Follow the exact format specified in the instructions
- If patterns or examples are provided, use them as guidance but create NEW content

Output ONLY a JSON array of dictionaries, nothing else.

JSON Array Output:"""

# Self-Instruct Distillation prompt for batch generation (generates multiple samples at once)
SELF_INSTRUCT_BATCH_GENERATION_PROMPT = """You are a DatasetGenerator. Your task is to generate high-quality training examples based on the provided demonstration examples.

## Task Instruction
{task_instruction}

{demo_examples_section}

## Your Task
Generate {batch_size} DIVERSE training examples as a JSON array.
Each example should be a dictionary with 'input' and 'output' keys.

IMPORTANT Requirements:
- Each example must be DISTINCT and cover DIFFERENT aspects of the task
- Ensure high diversity in topics, difficulty levels, and approaches
- Examples should be detailed, comprehensive, and high-quality
- Use the provided examples as guidance but create NEW content

Output ONLY a JSON array of dictionaries, nothing else.

JSON Array Output:"""

# Evol-Instruct Distillation prompt for batch generation (generates multiple samples at once)
EVOL_INSTRUCT_BATCH_GENERATION_IN_DEPTH_PROMPT = """You are an InstructionEvolver. Your goal is to perform IN-DEPTH EVOLUTION of training instructions.
Given the initial task and examples, you must rewrite or extend them into more COMPLEX and CHALLENGING versions while keeping their meaning coherent and reasonable.

## Task Instruction
{task_instruction}

{demo_examples_section}

## Your Task
Generate {batch_size} DIVERSE training examples as a JSON array.
Each example should be a dictionary with 'input' and 'output' keys.

IMPORTANT Requirements:
- Keep the same core topic and intention as the given examples
- Make it MORE COMPLEX in a reasonable way by applying ONE OR MORE of these strategies:
  1. **Add Constraints** – introduce additional requirements, rules, or limits
  2. **Deepen** – increase the scope or depth of the reasoning required
  3. **Concretize** – replace abstract descriptions with more specific or contextualized situations
  4. **Increase Reasoning Steps** – explicitly require multi-step logical thinking
  5. **Complicate Input** – provide structured data, code snippets, tables, or formulas as part of the input
- Ensure each evolved example is clear, human-understandable, and solvable
- Avoid redundancy or trivial rewording

Output ONLY a JSON array of dictionaries, nothing else.

JSON Array Output:"""

EVOL_INSTRUCT_BATCH_GENERATION_IN_BREADTH_PROMPT = """You are an InstructionEvolver. Your goal is to perform IN-BREADTH EVOLUTION of training instructions.
Given the initial task and examples, you must create NEW but RELATED instructions that explore different directions or perspectives of the same general domain.

## Task Instruction
{task_instruction}

{demo_examples_section}

## Your Task
Generate {batch_size} DIVERSE training examples as a JSON array.
Each example should be a dictionary with 'input' and 'output' keys.

IMPORTANT Requirements:
- Draw inspiration from the original task or examples
- Stay within the SAME general domain (topic, skill, or context)
- Create a DIFFERENT instruction that explores:
  - Another subtask or variation of the same problem type
  - A new angle, user intent, or application scenario
  - A different format (e.g., transformation, critique, generation, explanation)
- Ensure high topical DIVERSITY and creativity
- Keep difficulty and length roughly similar to the base examples
- All outputs must remain meaningful, well-formed, and answerable

Output ONLY a JSON array of dictionaries, nothing else.

JSON Array Output:"""


# ============================================================================
# PART4: IMAGE MODALITY PROMPTS
# ============================================================================

IMAGE_FIELD_FILTER_PROMPT = """You are a data field identifier for image-based QA datasets. Your task is to analyze the field names and determine which fields contain the image, input question, and output answer.

Given a list of field names from a dataset, identify:
- **image**: the field containing the image data (could be named "image", "img", "photo", "picture", etc.)
- **input**: the field containing the question/prompt about the image (could be named "question", "query", "prompt", "instruction", etc.)
- **output**: the field containing the answer/response (could be named "answer", "response", "label", "ground_truth", etc.)

### Output format:
Return a JSON object with three keys:
{{
  "image": "<name of the image field>",
  "input": "<name of the input/question field>",
  "output": "<name of the output/answer field>"
}}

### Rules:
1. Choose the field names that most likely correspond to image, question (input), and answer (output).
2. Only select from the provided legal keys.
3. If you cannot identify all three fields, use null for the missing ones:
   {{
     "image": null,
     "input": null,
     "output": null
   }}
4. Do NOT infer or fabricate field names not present in the list.
5. Only output the field names, not their contents.

### Example
Legal Keys: ["image", "question", "answer", "category"]

Output:
{{
  "image": "image",
  "input": "question",
  "output": "answer"
}}

### Example 2
Legal Keys: ["img", "prompt", "response", "metadata"]

Output:
{{
  "image": "img",
  "input": "prompt",
  "output": "response"
}}

Legal Keys: {legal_keys}

Output:
"""

IMAGE_META_PROMPT = """As a DatasetGenerator for image-based tasks, your task is to generate {num_samples} diverse question-answer pairs (`input` and `output`) based on the provided image.

IMPORTANT: All generated questions and answers MUST be directly based on the visual content of the provided image. The questions should ask about elements, details, or information that can only be answered by analyzing the image.

Try your best to ensure that the question-answer pairs you generate are distinct from each other while maintaining diverse, detailed, precise, comprehensive, and high-quality responses.

CRITICAL: You MUST strictly follow the output format instruction, especially any answer formatting requirements. The generated output must exactly match the specified format.
"""

