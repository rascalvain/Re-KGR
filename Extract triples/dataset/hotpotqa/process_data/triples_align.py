import json
import re
import time
from typing import List, Tuple, Dict, Optional
import os
from google import genai

# ================= 配置区域 =================
INPUT_FILE = "hotpot_dev_with_triples_filtered.json"
OUTPUT_FILE = "hotpot_dev_with_triples_aligned.json"
BATCH_SIZE = 15  # 每批处理的三元组数量
MAX_RECORDS = None  # 设置为 None 处理全部，设置数字(如 5)进行测试

# 🔥🔥🔥 核心开关 🔥🔥🔥
# True:  直接替换原始三元组（清洗模式）
# False: 保留原始数据，结果存入新字段（调试模式）
REPLACE_MODE = True

# API 限速配置
API_DELAY = 0.5  # 每次 API 调用之间的延迟（秒）
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 2  # 重试延迟（秒）

# ================= Gemini API 配置 =================
GEMINI_API_KEY = "sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN"
GEMINI_BASE_URL = "https://api.openai-proxy.org/google"
GEMINI_MODEL = "gemini-2.5-flash-lite"

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={"base_url": GEMINI_BASE_URL}
)

# ================= Prompt 设计 =================
PROMPT_TEMPLATE = """# System Role
You are a specialized Knowledge Graph Canonicalization Engine. Your task is to transform raw, unstructured triples into standardized, Wikidata-aligned triples suitable for knowledge graph construction.

# Objective
Given a batch of raw triples in the format [Subject, Relation, Object], you must:
- Standardize entity names using official Wikidata labels
- Map relations to standard Wikidata property labels or semantically equivalent predicates
- Ensure logical consistency by reordering triple components when necessary

# Detailed Processing Requirements
## Step 1: Entity Standardization (Subject & Object)
- Identify the real-world entity each string refers to
- Replace with the official English label from Wikidata
- Disambiguate when necessary using context clues
- Preserve entities that are already in standard form

Examples:
- "Jobs" → "Steve Jobs"
- "Apple" → "Apple Inc." (when referring to the company)
- "The Matrix" → "The Matrix" (already standard)
- "Obama" → "Barack Obama"

## Step 2: Relation Standardization
- Understand the semantic meaning of the raw relation
- Map to the corresponding Wikidata property label or a clear, standardized predicate
- Use consistent vocabulary across similar relations

Common Mappings:
- "started", "created", "established" → "founded by" or "founder"
- "is", "is a" → "instance of"
- "born on", "birth date" → "date of birth"
- "directed by" → "director"
- "worked on", "composed for" → "composer" or "contributor"
- "wife", "husband" → "spouse"
- "attended", "studied at" → "educated at"

## Step 3: Logical Direction Correction [CRITICAL]
This is the most important step. Many Wikidata properties have inherent directionality that may not match the raw triple.

Key Principle: The standardized triple must be logically TRUE according to how the standard relation is defined.

Decision Process:
- Examine the standardized relation's typical usage pattern
- Determine its canonical direction (e.g., "founded by" goes from Organization → Person)
- If the raw triple has the opposite direction, swap Subject and Object
- If directions match, maintain the original order

Critical Examples:
- Raw: ["Steve Jobs", "founded", "Apple"] → Standard: ["Apple Inc.", "founded by", "Steve Jobs"] ⚠️ SWAPPED
- Raw: ["Apple", "founder", "Steve Jobs"] → Standard: ["Apple Inc.", "founder", "Steve Jobs"] ✓ NO SWAP
- Raw: ["The Matrix", "directed by", "Wachowskis"] → Standard: ["The Matrix", "director", "Wachowskis"] ✓ NO SWAP
- Raw: ["Wachowskis", "directed", "The Matrix"] → Standard: ["The Matrix", "director", "Wachowskis"] ⚠️ SWAPPED
- Raw: ["Barack Obama", "spouse", "Michelle"] → Standard: ["Barack Obama", "spouse", "Michelle Obama"] ✓ NO SWAP

# Output Requirements
## Format
- Return ONLY a valid JSON array of arrays
- Each inner array must contain exactly 3 strings: [Subject, Relation, Object]
- NO explanatory text, comments, or markdown formatting
- NO Wikidata QIDs or PIDs - use labels only

## Quality Standards
- All entity names must be properly capitalized and complete
- Relations must be clear, standard English phrases
- Every triple must be logically coherent and factually plausible
- Maintain semantic equivalence with the original triple

# Examples
## Example 1: Basic Standardization
Input:
[
  ["Bill Gates", "created", "Microsoft"],
  ["Einstein", "born in", "Germany"],
  ["iPhone", "made by", "Apple"]
]

Output:
[
  ["Microsoft", "founded by", "Bill Gates"],
  ["Albert Einstein", "country of citizenship", "Germany"],
  ["iPhone", "manufacturer", "Apple Inc."]
]

## Example 2: Complex Direction Correction
Input:
[
  ["Elon Musk", "CEO of", "Tesla"],
  ["The Godfather", "star", "Marlon Brando"],
  ["Tyler Bates", "worked on", "Guardians of the Galaxy"]
]

Output:
[
  ["Tesla, Inc.", "chief executive officer", "Elon Musk"],
  ["The Godfather", "cast member", "Marlon Brando"],
  ["Guardians of the Galaxy", "composer", "Tyler Bates"]
]

## Example 3: Mixed Cases with Temporal Information
Input:
[
  ["Scott Derrickson", "directed", "Doctor Strange"],
  ["Doctor Strange", "release year", "2016"],
  ["Benedict Cumberbatch", "stars in", "Doctor Strange"]
]

Output:
[
  ["Doctor Strange", "director", "Scott Derrickson"],
  ["Doctor Strange", "publication date", "2016"],
  ["Doctor Strange", "cast member", "Benedict Cumberbatch"]
]

# Your Task
Process the following batch of raw triples. Apply all three steps carefully, paying special attention to logical direction correction.

# Input Batch:
{{BATCH_DATA}}

# Output (JSON only):"""


# ================= 统计信息 =================
class Statistics:
    def __init__(self):
        self.total_records = 0
        self.total_triples = 0
        self.aligned_triples = 0
        self.failed_triples = 0
        self.api_calls = 0
        self.start_time = time.time()

    def print_progress(self, current_record, total_records):
        elapsed = time.time() - self.start_time
        speed = self.aligned_triples / elapsed if elapsed > 0 else 0
        print(f"\n{'=' * 70}")
        print(f"📊 进度: {current_record}/{total_records} 记录")
        print(f"   - 已对齐三元组: {self.aligned_triples}")
        print(f"   - 失败三元组: {self.failed_triples}")
        print(f"   - API 调用次数: {self.api_calls}")
        print(f"   - 处理速度: {speed:.1f} 三元组/秒")
        print(f"   - 已用时间: {elapsed:.1f} 秒")


stats = Statistics()


# ================= 核心逻辑 =================

def parse_triple(triple_str: str) -> Optional[Tuple[str, str, str]]:
    """
    改进的三元组解析器，正确处理实体名称中的逗号
    支持格式: (Subject, Relation, Object)
    """
    content = triple_str.strip()

    # 移除首尾括号
    if content.startswith('(') and content.endswith(')'):
        content = content[1:-1]
    else:
        return None

    # 使用状态机解析，正确处理嵌套的逗号
    parts = []
    current_part = ""
    depth = 0  # 括号/引号深度
    in_quotes = False

    i = 0
    while i < len(content):
        char = content[i]

        if char == '"':
            in_quotes = not in_quotes
            current_part += char
        elif in_quotes:
            current_part += char
        elif char in '([':
            depth += 1
            current_part += char
        elif char in ')]':
            depth -= 1
            current_part += char
        elif char == ',' and depth == 0:
            # 只在顶层逗号处分割
            parts.append(current_part.strip())
            current_part = ""
            # 跳过逗号后的空格
            while i + 1 < len(content) and content[i + 1] == ' ':
                i += 1
        else:
            current_part += char

        i += 1

    # 添加最后一部分
    if current_part:
        parts.append(current_part.strip())

    # 验证是否有恰好三个部分且都非空
    if len(parts) == 3 and all(p for p in parts):
        return tuple(parts)

    return None


def call_llm(prompt_text: str) -> str:
    """调用 Gemini API，带重试机制"""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt_text
            )
            stats.api_calls += 1
            return response.text
        except Exception as e:
            print(f"      ⚠️ API 错误 (尝试 {attempt + 1}/{MAX_RETRIES}): {str(e)[:100]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return "[]"
    return "[]"


def clean_response(response_text: str) -> str:
    """清理 LLM 响应，移除 markdown 标记"""
    cleaned = response_text.strip()
    # 移除 markdown 代码块标记
    cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^```\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()


def process_batch_triples(raw_tuples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    处理一批三元组，返回对齐后的结果
    如果处理失败，返回原始数据
    """
    if not raw_tuples:
        return []

    # 准备输入
    input_data = [[s, r, o] for s, r, o in raw_tuples]
    input_json = json.dumps(input_data, ensure_ascii=False, indent=2)
    prompt = PROMPT_TEMPLATE.replace("{{BATCH_DATA}}", input_json)

    # 调用 API
    response = call_llm(prompt)

    # 解析响应
    try:
        cleaned = clean_response(response)
        result = json.loads(cleaned)

        # 验证格式
        if not isinstance(result, list):
            print(f"      ⚠️ 响应不是列表格式")
            return raw_tuples

        # 验证每个元素
        valid_results = []
        for item in result:
            if isinstance(item, list) and len(item) == 3:
                valid_results.append(tuple(item))
            else:
                print(f"      ⚠️ 无效的三元组格式: {item}")

        # 如果结果数量不匹配，返回原始数据
        if len(valid_results) != len(raw_tuples):
            print(f"      ⚠️ 结果数量不匹配: 期望 {len(raw_tuples)}, 得到 {len(valid_results)}")
            return raw_tuples

        return valid_results

    except json.JSONDecodeError as e:
        print(f"      ⚠️ JSON 解析失败: {e}")
        print(f"      响应前 200 字符: {response[:200]}")
        return raw_tuples
    except Exception as e:
        print(f"      ⚠️ 处理失败: {e}")
        return raw_tuples


def process_triple_list(triple_objects: List[Dict], list_type: str) -> List[Dict]:
    """
    处理单条数据中的三元组列表
    """
    if not triple_objects:
        return []

    # 1. 解析所有三元组
    parsed_data = []  # [(index, original_obj, parsed_tuple)]

    for i, obj in enumerate(triple_objects):
        triple_str = obj.get('triple', '')
        parsed = parse_triple(triple_str)

        if parsed:
            parsed_data.append((i, obj, parsed))
        else:
            print(f"      ⚠️ 跳过无效三元组: {triple_str[:60]}...")

    if not parsed_data:
        print(f"      ✗ 没有有效的三元组")
        return triple_objects

    print(f"      ✓ 解析成功: {len(parsed_data)}/{len(triple_objects)}")

    # 2. 批量处理
    aligned_results = []
    total_batches = (len(parsed_data) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(parsed_data), BATCH_SIZE):
        batch_data = parsed_data[batch_idx:batch_idx + BATCH_SIZE]
        batch_tuples = [item[2] for item in batch_data]
        batch_num = batch_idx // BATCH_SIZE + 1

        print(f"      📦 批次 {batch_num}/{total_batches} ({len(batch_tuples)} 个)...", end=" ")

        # 调用对齐
        aligned_tuples = process_batch_triples(batch_tuples)
        aligned_results.extend(aligned_tuples)

        # 统计
        success_count = sum(1 for orig, aligned in zip(batch_tuples, aligned_tuples) if orig != aligned)
        stats.aligned_triples += len(aligned_tuples)

        print(f"✓ ({success_count} 个已对齐)")

        # API 限速
        time.sleep(API_DELAY)

    # 3. 构造结果
    result_list = []
    aligned_idx = 0

    for i, original_obj in enumerate(triple_objects):
        # 查找是否在 parsed_data 中
        parsed_item = next((item for item in parsed_data if item[0] == i), None)

        if parsed_item and aligned_idx < len(aligned_results):
            # 使用对齐后的结果
            sub, rel, obj = aligned_results[aligned_idx]
            aligned_idx += 1

            if REPLACE_MODE:
                # 替换模式：只保留 triple 字段
                new_obj = {"triple": f"({sub}, {rel}, {obj})"}
            else:
                # 保留模式：保留原始数据并添加对齐信息
                new_obj = original_obj.copy()
                new_obj['aligned_triple'] = f"({sub}, {rel}, {obj})"
                new_obj['subject_aligned'] = sub
                new_obj['relation_aligned'] = rel
                new_obj['object_aligned'] = obj

            result_list.append(new_obj)
        else:
            # 无法解析或对齐的，保留原样
            result_list.append(original_obj)
            stats.failed_triples += 1

    return result_list


def main():
    """主函数"""
    print("=" * 70)
    print("🚀 三元组对齐与消歧工具 (Gemini API - 优化版)")
    print("=" * 70)
    print(f"📋 模式: {'🔥 替换原数据' if REPLACE_MODE else '✅ 保留原数据'}")
    print(f"🤖 模型: {GEMINI_MODEL}")
    print(f"📦 批次大小: {BATCH_SIZE}")
    print(f"⏱️ API 延迟: {API_DELAY}s")

    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ 错误: 输入文件不存在: {INPUT_FILE}")
        return

    # 读取数据
    print(f"\n📂 读取文件: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 确定处理范围
    target_data = data[:MAX_RECORDS] if MAX_RECORDS else data
    stats.total_records = len(target_data)

    print(f"📊 数据集大小: {len(data)} 条记录")
    if MAX_RECORDS:
        print(f"⚠️ 测试模式: 仅处理前 {MAX_RECORDS} 条")

    # 处理每条记录
    try:
        for idx, record in enumerate(target_data):
            record_id = record.get('_id', 'N/A')
            print(f"\n{'=' * 70}")
            print(f"📄 记录 [{idx + 1}/{stats.total_records}]: {record_id}")

            # 处理 context_triples
            if 'context_triples' in record and record['context_triples']:
                print(f"  🔹 context_triples ({len(record['context_triples'])} 个)")
                processed = process_triple_list(record['context_triples'], 'context')

                if REPLACE_MODE:
                    record['context_triples'] = processed
                else:
                    record['context_triples_aligned'] = processed

            # 处理 gpt_sentence_triples
            if 'gpt_sentence_triples' in record and record['gpt_sentence_triples']:
                print(f"  🔹 gpt_sentence_triples ({len(record['gpt_sentence_triples'])} 个)")
                processed = process_triple_list(record['gpt_sentence_triples'], 'gpt')

                if REPLACE_MODE:
                    record['gpt_sentence_triples'] = processed
                else:
                    record['gpt_sentence_triples_aligned'] = processed

            # 每10条保存一次
            if (idx + 1) % 10 == 0:
                print(f"\n  💾 中间保存...")
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(target_data, f, indent=2, ensure_ascii=False)
                stats.print_progress(idx + 1, stats.total_records)

        # 最终保存
        print(f"\n{'=' * 70}")
        print(f"💾 保存最终结果到: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(target_data, f, indent=2, ensure_ascii=False)

        # 最终统计
        elapsed = time.time() - stats.start_time
        print(f"\n{'=' * 70}")
        print(f"✅ 处理完成！")
        print(f"📊 最终统计:")
        print(f"   - 处理记录: {stats.total_records}")
        print(f"   - 对齐三元组: {stats.aligned_triples}")
        print(f"   - 失败三元组: {stats.failed_triples}")
        print(f"   - API 调用: {stats.api_calls}")
        print(f"   - 总耗时: {elapsed:.1f} 秒")
        print(f"   - 平均速度: {stats.aligned_triples / elapsed:.1f} 三元组/秒")
        print(f"   - 估算成本: ~${stats.api_calls * 0.001:.4f} (假设每次调用 $0.001)")
        print(f"{'=' * 70}")

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断！正在保存已处理的数据...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(target_data, f, indent=2, ensure_ascii=False)
        print(f"💾 已保存到: {OUTPUT_FILE}")
    except Exception as e:
        print(f"\n❌ 处理出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()