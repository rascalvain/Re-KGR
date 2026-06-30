from openai import OpenAI
import openai
import time
import json
import string
import random
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util

client = OpenAI(
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN",
)

MODEL = "gpt-3.5-turbo"

# ---------- 语义相似度匹配配置 ----------
SEMANTIC_MODEL_NAME = "all-mpnet-base-v2"
SEMANTIC_THRESHOLD = 0.75

print(f"正在加载语义匹配模型: {SEMANTIC_MODEL_NAME} ...")
from transformers import AutoModel, AutoConfig
_orig_from_pretrained = AutoModel.from_pretrained
AutoModel.from_pretrained = lambda *args, **kwargs: _orig_from_pretrained(
    *args, **{**kwargs, "ignore_mismatched_sizes": True}
)
_semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
AutoModel.from_pretrained = _orig_from_pretrained
print("语义匹配模型加载完成。")

# ========================================================================== #
#  Prompt 模板
# ========================================================================== #

# ---------- 第一阶段：正常生成正确推理链 ----------
ANSWER_PROMPT_NO_CONTEXT = """You are a knowledgeable assistant. Please answer the following question based on your knowledge.

Question: {question}

For the following question, reason step by step and end with "So the answer is: ...".

IMPORTANT:
- Provide your reasoning process step by step
- After your reasoning, end with "So the answer is: X" where X is a SHORT, CONCISE answer
- The final answer should be brief (e.g., "yes", "no", a name, a date, a number, a short phrase)
- DO NOT write a full sentence as the final answer

Please provide your reasoning process, then give a short final answer."""

ANSWER_PROMPT_WITH_ENTITIES = """You are a knowledgeable assistant. The following entities may be relevant to the question:

Relevant entities: {entities}

Question: {question}

For the following question, reason step by step and end with "So the answer is: ...".

IMPORTANT:
- Provide your reasoning process step by step
- After your reasoning, end with "So the answer is: X" where X is a SHORT, CONCISE answer
- The final answer should be brief (e.g., "yes", "no", a name, a date, a number, a short phrase)
- DO NOT write a full sentence as the final answer

Please provide your reasoning process, then give a short final answer."""

# ---------- 第二阶段：对正确推理链进行事实篡改，制造含幻觉推理过程 ----------
HALLUCINATION_REWRITE_PROMPT = """Below is a correct reasoning chain for a question. Your task is to rewrite it by introducing 1-2 subtle factual errors into the reasoning process, which should naturally lead to a different (incorrect) final answer.

Original question: {question}
Correct answer: {correct_answer}
Original correct reasoning:
{correct_reasoning}

Requirements:
- Modify 1-2 specific facts in the reasoning (e.g., change a name, date, number, relationship, or attribute to a similar but wrong one)
- The modified reasoning must still read fluently, sound convincing, and appear natural
- The factual errors should be SUBTLE and PLAUSIBLE (e.g., confuse with a similar entity, off-by-one number, nearby date, related but wrong person)
- The final answer MUST be DIFFERENT from "{correct_answer}"
- Update the final answer to logically follow from the corrupted reasoning
- Keep the same format: reasoning steps followed by "So the answer is: X"
- Do NOT add any disclaimer, note, or hint that facts were changed
- Do NOT mention the original correct answer anywhere
- Write as if you genuinely believe the wrong facts

Rewritten reasoning with subtle factual errors:"""


# ========================================================================== #
#  工具函数
# ========================================================================== #

def load_progress(save_path):
    """加载已保存的进度"""
    try:
        with open(save_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_data(data, save_path):
    """保存数据"""
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def request_api(prompt, model, temperature, max_tokens=500):
    """调用 OpenAI API，支持重试与限速处理"""
    retry_count = 0
    max_retries = 5

    while retry_count < max_retries:
        try:
            message = [{'role': 'user', 'content': prompt}]
            api_params = {
                "model": model,
                "messages": message,
                "n": 1
            }

            try:
                api_params["max_tokens"] = max_tokens
                response = client.chat.completions.create(**api_params)
            except (TypeError, openai.BadRequestError):
                print("  尝试使用 max_completion_tokens 参数...")
                api_params.pop("max_tokens", None)
                api_params["max_completion_tokens"] = max_tokens * 2
                response = client.chat.completions.create(**api_params)

            return response.choices[0].message.content.strip()

        except openai.RateLimitError:
            wait_time = 2 ** retry_count
            print(f"  速率限制超出，等待 {wait_time} 秒...")
            time.sleep(wait_time)
            retry_count += 1

        except openai.APIError as e:
            print(f"  API 错误: {e}")
            if "timeout" in str(e).lower():
                time.sleep(2)
            retry_count += 1

        except Exception as e:
            print(f"  API调用错误: {type(e).__name__}: {e}")
            time.sleep(1)
            retry_count += 1

    raise Exception(f"API调用失败，超过最大重试次数 ({max_retries})")


# ========================================================================== #
#  数据集字段提取
# ========================================================================== #

def extract_supporting_entity_labels(answer_obj):
    """从 answer 对象中提取 supportingEnt 的英文标签列表"""
    labels = []
    for ent in answer_obj.get("supportingEnt", []):
        label = ent.get("label", {})
        if isinstance(label, dict):
            en_label = label.get("en")
            if en_label:
                labels.append(en_label)
        elif isinstance(label, str):
            labels.append(label)
    return labels


def extract_question_entity_labels(question_entities):
    """从 questionEntity 列表中提取实体标签"""
    labels = []
    for ent in question_entities:
        if ent.get("entityType", "") == "entity":
            label = ent.get("label")
            if label:
                labels.append(label)
    return labels


def get_correct_answer_str(answer_obj):
    """从 Mintaka answer 对象中提取标准答案字符串

    返回 (mention_str, [候选答案列表])
    注意：不包含 Wikidata ID（如 Q12345），避免短答案误匹配。
    """
    mention = answer_obj.get("mention", "")
    answer_type = answer_obj.get("answerType", "")
    answers = answer_obj.get("answer", [])

    candidates = []

    # mention 可能是逗号分隔的多答案，拆开后每个都作为独立候选
    if mention:
        candidates.append(str(mention))
        if "," in mention:
            for part in mention.split(","):
                part = part.strip()
                if part:
                    candidates.append(part)

    if answer_type == "entity":
        for ans in answers:
            if isinstance(ans, dict):
                label = ans.get("label", {})
                if isinstance(label, dict):
                    en = label.get("en")
                    if en:
                        candidates.append(en)
                elif isinstance(label, str):
                    candidates.append(label)
                # 不加入 Wikidata ID（如 Q12345），避免短答案子串误匹配
    elif answer_type in ("numerical", "boolean", "date", "string"):
        for ans in answers:
            candidates.append(str(ans))

    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return mention, unique


# ========================================================================== #
#  答案比较
# ========================================================================== #

def normalize_answer(answer):
    """标准化答案：转小写、去标点、去多余空格"""
    if answer is None:
        return ""
    answer = str(answer).lower().strip()
    answer = ' '.join(answer.split())
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    return answer


def _contains_as_word(haystack, needle):
    """检查 needle 是否作为完整词出现在 haystack 中（按空格分词）"""
    hay_words = haystack.split()
    ndl_words = needle.split()
    ndl_len = len(ndl_words)
    for i in range(len(hay_words) - ndl_len + 1):
        if hay_words[i:i + ndl_len] == ndl_words:
            return True
    return False


def semantic_similarity(text_a, text_b):
    """使用 sentence-transformers 计算两段文本的余弦相似度，返回 float"""
    embeddings = _semantic_model.encode([text_a, text_b], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return score


MIN_CONTAIN_LEN = 4

def is_answer_correct(generated_answer, correct_candidates):
    """判断生成的答案是否与任一候选答案匹配

    三级匹配规则（依次尝试，命中即返回 True）：
    1. 标准化后完全相等
    2. sentence-transformer 语义相似度 ≥ SEMANTIC_THRESHOLD
    3. 较短串 ≥ MIN_CONTAIN_LEN 字符，且作为完整词组出现在较长串中
    """
    gen_norm = normalize_answer(generated_answer)
    if not gen_norm:
        return False

    # 规则 1：标准化后完全匹配
    for candidate in correct_candidates:
        cor_norm = normalize_answer(candidate)
        if not cor_norm:
            continue
        if gen_norm == cor_norm:
            return True

    # 规则 2：语义相似度匹配
    valid_candidates = [c for c in correct_candidates if normalize_answer(c)]
    if valid_candidates:
        gen_text = str(generated_answer).strip()
        scores = [semantic_similarity(gen_text, c.strip()) for c in valid_candidates]
        max_score = max(scores)
        best_candidate = valid_candidates[scores.index(max_score)]
        if max_score >= SEMANTIC_THRESHOLD:
            print(f"    [语义匹配] \"{gen_text}\" ↔ \"{best_candidate}\" "
                  f"相似度={max_score:.3f} ≥ {SEMANTIC_THRESHOLD} → 匹配")
            return True

    # 规则 3：完整词包含匹配
    for candidate in correct_candidates:
        cor_norm = normalize_answer(candidate)
        if not cor_norm:
            continue
        shorter, longer = (gen_norm, cor_norm) if len(gen_norm) <= len(cor_norm) else (cor_norm, gen_norm)
        if len(shorter) >= MIN_CONTAIN_LEN and _contains_as_word(longer, shorter):
            return True

    return False


def extract_final_answer(gpt_response):
    """从 GPT 响应中提取最终答案（"So the answer is:" 之后的部分）"""
    if "So the answer is:" in gpt_response:
        final_answer = gpt_response.split("So the answer is:")[-1].strip()
        final_answer = final_answer.rstrip('.,!?;:"\' ')
        final_answer = final_answer.lstrip('"\' ')
        return final_answer
    return None


# ========================================================================== #
#  两阶段生成核心逻辑
# ========================================================================== #

def build_entity_str(answer_obj, question_entities):
    """构建实体提示字符串"""
    supporting = extract_supporting_entity_labels(answer_obj)
    q_entities = extract_question_entity_labels(question_entities)
    all_labels = list(dict.fromkeys(q_entities + supporting))
    return ", ".join(all_labels) if all_labels else ""


def generate_correct_answer(question, answer_obj, question_entities, temperature=0.1):
    """第一阶段：用低温度生成正确推理链"""
    entity_str = build_entity_str(answer_obj, question_entities)

    if entity_str:
        prompt = ANSWER_PROMPT_WITH_ENTITIES.format(question=question, entities=entity_str)
    else:
        prompt = ANSWER_PROMPT_NO_CONTEXT.format(question=question)

    return request_api(prompt, model=MODEL, temperature=temperature)


def generate_hallucinated_rewrite(question, correct_answer, correct_reasoning, temperature=0.9):
    """第二阶段：对正确推理链进行事实篡改，生成含幻觉的推理过程"""
    prompt = HALLUCINATION_REWRITE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        correct_reasoning=correct_reasoning
    )
    return request_api(prompt, model=MODEL, temperature=temperature, max_tokens=600)


def process_single_correct(item, correct_temp):
    """处理一条「正确」样本：低温度生成，验证匹配后保留

    最多重试 MAX_CORRECT_RETRIES 次以获得正确答案。
    若 3 次均未匹配，则将最后一次生成结果作为幻觉数据保留。

    返回 (result_item, actual_label: str)
        actual_label 为 'correct' 或 'hallucination'（自然产生的幻觉）
    """
    MAX_CORRECT_RETRIES = 3
    question = item.get('question', '')
    answer_obj = item.get('answer', {})
    question_entities = item.get('questionEntity', [])
    correct_mention, correct_candidates = get_correct_answer_str(answer_obj)

    last_response = None
    last_final_answer = None

    for attempt in range(MAX_CORRECT_RETRIES):
        gpt_response = generate_correct_answer(
            question, answer_obj, question_entities, temperature=correct_temp
        )
        final_answer = extract_final_answer(gpt_response)
        last_response = gpt_response
        last_final_answer = final_answer

        if is_answer_correct(final_answer, correct_candidates):
            result = item.copy()
            result['gpt_sentence'] = gpt_response
            result['gpt_final_answer'] = final_answer
            result['generation_label'] = 'correct'
            result['model_used'] = MODEL
            result['temperature'] = correct_temp
            result['generation_mode'] = 'correct_direct'
            return result, 'correct'

        if attempt < MAX_CORRECT_RETRIES - 1:
            print(f"    正确组重试 ({attempt + 1}/{MAX_CORRECT_RETRIES}): "
                  f"生成 [{final_answer}] ≠ 期望 [{correct_mention}]")
            time.sleep(0.3)

    # 3 次均未匹配正确答案 → 自然产生的幻觉，保留为 hallucination
    print(f"    ⚠️ 正确组 {MAX_CORRECT_RETRIES} 次均未匹配，回退保留为幻觉数据")
    result = item.copy()
    result['gpt_sentence'] = last_response
    result['gpt_final_answer'] = last_final_answer
    result['generation_label'] = 'hallucination'
    result['model_used'] = MODEL
    result['temperature'] = correct_temp
    result['generation_mode'] = 'correct_fallback_to_hallucination'
    return result, 'hallucination'


def process_single_hallucination(item, correct_temp, hallucination_temp):
    """处理一条「幻觉」样本：两阶段生成

    阶段1：低温度生成正确推理链
    阶段2：用篡改 prompt 改写推理链，嵌入幻觉事实
    验证篡改后答案确实与正确答案不同。

    若篡改 3 次均失败（篡改后答案仍匹配正确答案），则将阶段1的
    正确推理链作为 correct 数据保留，不浪费 API 调用。

    返回 (result_item, actual_label: str)
        actual_label 为 'hallucination' 或 'correct'（篡改失败回退）
    """
    MAX_HALLUCINATION_RETRIES = 3
    question = item.get('question', '')
    answer_obj = item.get('answer', {})
    question_entities = item.get('questionEntity', [])
    correct_mention, correct_candidates = get_correct_answer_str(answer_obj)

    # 阶段1：生成正确推理链
    correct_reasoning = generate_correct_answer(
        question, answer_obj, question_entities, temperature=correct_temp
    )
    correct_final = extract_final_answer(correct_reasoning)
    print(f"    阶段1 正确推理答案: [{correct_final}]")

    # 阶段2：篡改推理链，嵌入幻觉事实
    for attempt in range(MAX_HALLUCINATION_RETRIES):
        hallucinated_response = generate_hallucinated_rewrite(
            question=question,
            correct_answer=correct_mention or (correct_final or ""),
            correct_reasoning=correct_reasoning,
            temperature=hallucination_temp
        )
        hallucinated_final = extract_final_answer(hallucinated_response)

        # 验证篡改后答案确实不同
        if hallucinated_final and not is_answer_correct(hallucinated_final, correct_candidates):
            result = item.copy()
            result['gpt_sentence'] = hallucinated_response
            result['gpt_final_answer'] = hallucinated_final
            result['generation_label'] = 'hallucination'
            result['correct_reasoning'] = correct_reasoning
            result['correct_final_answer'] = correct_final
            result['model_used'] = MODEL
            result['temperature'] = hallucination_temp
            result['generation_mode'] = 'two_stage_rewrite'
            return result, 'hallucination'

        if attempt < MAX_HALLUCINATION_RETRIES - 1:
            print(f"    幻觉组篡改重试 ({attempt + 1}/{MAX_HALLUCINATION_RETRIES}): "
                  f"篡改答案 [{hallucinated_final}] 仍匹配正确答案，需重新篡改")
            time.sleep(0.3)

    # 篡改 3 次均失败 → 回退为正确数据，不浪费已有的阶段1结果
    print(f"    ⚠️ 篡改 {MAX_HALLUCINATION_RETRIES} 次均失败，回退保留为正确数据")
    result = item.copy()
    result['gpt_sentence'] = correct_reasoning
    result['gpt_final_answer'] = correct_final
    result['generation_label'] = 'correct'
    result['model_used'] = MODEL
    result['temperature'] = correct_temp
    result['generation_mode'] = 'hallucination_fallback_to_correct'
    return result, 'correct'


# ========================================================================== #
#  主处理流程
# ========================================================================== #

def process_mintaka_balanced(input_path, output_path,
                             correct_temp=0.1, hallucination_temp=0.9,
                             target_samples=None):
    """两阶段平衡生成：正确 / 幻觉各占 50%

    流程：
    1. 加载数据，随机打乱后前一半分配给正确组，后一半分配给幻觉组
    2. 正确组：低温度生成 → 验证匹配 → 保留
    3. 幻觉组：低温度生成正确推理 → 篡改推理链嵌入幻觉事实 → 验证不匹配 → 保留
    4. 交替处理两组，实时保存进度
    """
    print(f"正在加载数据文件: {input_path}")
    print(f"使用模型: {MODEL}")
    print(f"正确组 Temperature: {correct_temp}")
    print(f"幻觉组 Temperature: {hallucination_temp}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    # 限制样本数
    if target_samples and target_samples < len(data):
        random.seed(42)
        data = random.sample(data, target_samples)
    total = len(data)

    # 随机打乱后分成两组
    random.seed(42)
    indices = list(range(total))
    random.shuffle(indices)
    half = total // 2
    correct_indices = set(indices[:half])
    hallucination_indices = set(indices[half:])

    print(f"总数据量: {total}")
    print(f"正确组目标: {len(correct_indices)} 条 | 幻觉组目标: {len(hallucination_indices)} 条")

    # 断点续传
    processed_data = load_progress(output_path)
    processed_ids = {item.get('id') for item in processed_data}
    correct_count = sum(1 for it in processed_data if it.get('generation_label') == 'correct')
    hallucination_count = sum(1 for it in processed_data if it.get('generation_label') == 'hallucination')
    skip_count = 0

    if processed_data:
        print(f"已有进度: {len(processed_data)} 条 "
              f"(正确 {correct_count} / 幻觉 {hallucination_count})，继续处理...")

    start_time = time.time()
    fallback_count = sum(1 for it in processed_data
                         if it.get('generation_mode') in (
                             'hallucination_fallback_to_correct',
                             'correct_fallback_to_hallucination'))

    for idx in range(total):
        item = data[idx]
        item_id = item.get('id', str(idx))

        if item_id in processed_ids:
            skip_count += 1
            continue

        target_label = "correct" if idx in correct_indices else "hallucination"
        question = item.get('question', '')
        answer_obj = item.get('answer', {})
        correct_mention, _ = get_correct_answer_str(answer_obj)
        answer_type = answer_obj.get('answerType', 'unknown')
        complexity = item.get('complexityType', 'unknown')
        category = item.get('category', 'unknown')

        processed_so_far = len(processed_data) - (correct_count + hallucination_count - correct_count - hallucination_count) + 1
        print(f"\n{'=' * 80}")
        print(f"[{len(processed_data) + 1}/{total}]  ID: {item_id}  目标: {target_label.upper()}")
        print(f"类别: {category} | 复杂度: {complexity} | 答案类型: {answer_type}")
        print(f"问题: {question}")
        print(f"正确答案: {correct_mention}")

        try:
            if target_label == "correct":
                result, actual_label = process_single_correct(item, correct_temp)
            else:
                result, actual_label = process_single_hallucination(item, correct_temp, hallucination_temp)

            label = actual_label
            if actual_label == 'correct':
                correct_count += 1
                if result.get('generation_mode') == 'hallucination_fallback_to_correct':
                    fallback_count += 1
            else:
                hallucination_count += 1
                if result.get('generation_mode') == 'correct_fallback_to_hallucination':
                    fallback_count += 1
            processed_data.append(result)

            print(f"\n生成的完整回答:")
            print(f"{result['gpt_sentence']}")
            print(f"\n{'─' * 40}")
            print(f"提取的最终答案: [{result['gpt_final_answer']}]")
            print(f"数据集答案:     [{correct_mention}]")
            mode_desc = {'correct_direct': '✓ 正确',
                         'two_stage_rewrite': '✗ 含幻觉推理',
                         'hallucination_fallback_to_correct': '✓ 正确（篡改失败回退）',
                         'correct_fallback_to_hallucination': '✗ 幻觉（自然产生）'}
            print(f"标签: {label} ({mode_desc.get(result.get('generation_mode', ''), label)})")

            # 每 5 条保存一次
            if len(processed_data) % 5 == 0 and len(processed_data) > 0:
                elapsed = time.time() - start_time
                n_done = len(processed_data) - (len(processed_ids) if skip_count else 0)
                avg_time = elapsed / max(n_done - skip_count, 1)
                remaining_count = total - idx - 1
                remaining_time = remaining_count * avg_time

                c_ratio = correct_count / len(processed_data) * 100
                h_ratio = hallucination_count / len(processed_data) * 100

                print(f"\n💾 保存进度... ({len(processed_data)} 条)")
                print(f"   正确 {correct_count} ({c_ratio:.1f}%) | 幻觉 {hallucination_count} ({h_ratio:.1f}%)")
                print(f"   其中回退（标签与目标不同）: {fallback_count} 条")
                print(f"   预计剩余: {remaining_time / 60:.1f} 分钟")
                save_data(processed_data, output_path)

            time.sleep(0.2)

        except Exception as e:
            print(f"❌ 处理出错: {e}")
            import traceback
            traceback.print_exc()
            save_data(processed_data, output_path)
            print("  将在 5 秒后继续...")
            time.sleep(5)
            continue

    # 最终保存
    save_data(processed_data, output_path)
    total_time = time.time() - start_time
    current_total = len(processed_data)

    print(f"\n{'=' * 80}")
    print(f"处理完成！结果已保存到: {output_path}")
    fb_h2c = sum(1 for it in processed_data if it.get('generation_mode') == 'hallucination_fallback_to_correct')
    fb_c2h = sum(1 for it in processed_data if it.get('generation_mode') == 'correct_fallback_to_hallucination')

    print(f"\n📊 最终统计:")
    print(f"  - 总数据量:   {current_total}")
    print(f"  - 正确答案:   {correct_count} ({correct_count / max(current_total, 1) * 100:.1f}%)")
    print(f"  - 幻觉答案:   {hallucination_count} ({hallucination_count / max(current_total, 1) * 100:.1f}%)")
    print(f"  - 回退明细:")
    print(f"      幻觉组篡改失败 → 正确: {fb_h2c} 条")
    print(f"      正确组生成失败 → 幻觉: {fb_c2h} 条")
    print(f"  - 总耗时:     {total_time / 60:.1f} 分钟")
    print(f"  - 使用模型:   {MODEL}")
    print(f"  - 正确组温度: {correct_temp}")
    print(f"  - 幻觉组温度: {hallucination_temp}")

    return processed_data


# ========================================================================== #
#  入口
# ========================================================================== #

if __name__ == "__main__":
    SPLIT = "dev"
    input_file = f"../mintaka-main/data/mintaka_{SPLIT}.json"
    output_file = f"data/mintaka_{SPLIT}_with_gpt_answers.json"

    # 正确组用低温度保证准确，幻觉组篡改时用较高温度增加多样性
    CORRECT_TEMP = 0.1
    HALLUCINATION_TEMP = 0.9

    # 目标样本数（两组各占一半）；None 表示处理全部
    TARGET_SAMPLES = None

    os.makedirs("data", exist_ok=True)

    print(f"""
{'=' * 80}
Mintaka 数据集处理脚本 — 两阶段平衡生成版（正确 50% / 幻觉 50%）
{'=' * 80}
配置信息:
  - 数据分割:     {SPLIT}
  - 输入文件:     {input_file}
  - 输出文件:     {output_file}
  - 模型:         {MODEL}
  - 正确组温度:   {CORRECT_TEMP}
  - 幻觉组温度:   {HALLUCINATION_TEMP}
  - 目标样本数:   {'全部' if TARGET_SAMPLES is None else TARGET_SAMPLES}
  - 生成策略:
      正确组 → 低温度直接生成，验证答案匹配
      幻觉组 → 阶段1 低温度生成正确推理链
             → 阶段2 篡改推理链嵌入幻觉事实，验证答案不匹配
{'=' * 80}
    """)

    try:
        result = process_mintaka_balanced(
            input_file, output_file,
            correct_temp=CORRECT_TEMP,
            hallucination_temp=HALLUCINATION_TEMP,
            target_samples=TARGET_SAMPLES
        )
        print(f"\n✅ 成功处理 {len(result)} 条数据")

        correct = sum(1 for it in result if it.get('generation_label') == 'correct')
        hallucination = sum(1 for it in result if it.get('generation_label') == 'hallucination')
        print(f"\n标签分布:")
        print(f"  正确: {correct} ({correct / max(len(result), 1) * 100:.1f}%)")
        print(f"  幻觉: {hallucination} ({hallucination / max(len(result), 1) * 100:.1f}%)")

    except KeyboardInterrupt:
        print("\n⏸️  用户中断，进度已保存")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
