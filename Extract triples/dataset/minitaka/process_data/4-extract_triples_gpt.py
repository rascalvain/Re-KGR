"""
Mintaka 数据集处理 — 第 3 步：从 GPT 推理文本中提取知识三元组

参考 HotpotQA 版本 (3-extract_triples_hotpot.py) 的两阶段提取流程：

  阶段 1  初始提取 (Triple Extraction)
      用 GPT_TRIPLE_EXTRACTION_PROMPT 提示 LLM，从 gpt_sentence
      中提取所有知识三元组，格式为 ("head", "relation", "tail")。

  阶段 2  验证修正 (Triple Revision)
      将阶段 1 的结果连同原文再次送入 LLM，修正语义错误、
      代词指代、关系表述等问题，输出最终三元组列表。

与 HotpotQA 不同之处：
  - Mintaka 没有 context 字段，仅对 gpt_sentence 提取
  - 三元组以结构化字典 {"head", "relation", "tail"} 存储
    （与 Wikidata entity_triples 中单条三元组格式完全一致）
  - 输出字段名 gpt_triples（与步骤 4/5 过滤清理脚本保持一致）

输入：第 2 步过滤输出 mintaka_dev_with_answers_filtered.json
输出：mintaka_dev_with_gpt_triples.json
"""

import json
import time
import os
import re
from openai import OpenAI
import openai

# ========================================================================== #
#  配置
# ========================================================================== #

client = OpenAI(
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN",
)
LLM_MODEL = "gpt-3.5-turbo"

# ========================================================================== #
#  Prompt
# ========================================================================== #

GPT_TRIPLE_EXTRACTION_PROMPT = """\
In the knowledge graph, knowledge triples are a basic data structure used to represent and store information, and each triple is an expression of a fact. Given a piece of text, please extract all knowledge triples contained in the text, and represent the triples in the form of ("head entity", "relationship", "tail entity").
Note that the extracted triples need to be as fine-grained as possible. It is necessary to ensure that the semantics of the triple are consistent with the information in the corresponding part of the text and there is not a pronoun in the triple. All knowledge triples in the text need to be extracted.

Here is an in-context example:
<text>: Paris, the capital of France, is a city with a long history and full of romance. Not only is there the world-famous Eiffel Tower and Louvre Museum, but it also has a unique artistic atmosphere and rich cultural heritage.
<response>:
Triple: (Paris, is, the capital of France)
Triple: (Paris, possession, long history)
Triple: (Paris, full, romantic)
Triple: (Paris, possession, Eiffel Tower)
Triple: (Paris, possession, Louvre)
Triple: (Paris, possessions, unique artistic atmosphere)
Triple: (Paris, possessions, rich cultural heritage)

<text>: {init_text}
<response>:"""

GPT_TRIPLE_EXTRACTION_PROMPT_REVISE = """\
Below are the knowledge Triples you extracted based on the text, but there are still some errors in it. For example, the semantics of the triple are different from the semantics of the corresponding part in the original text or there is a pronoun in the triple. Please check and correct.
<initial prompt>{p}
<triples>{t}
Please output all corrected triples directly, including changed and unmodified ones. Don't output any other words."""


# ========================================================================== #
#  工具函数
# ========================================================================== #

def load_progress(save_path):
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_data(data, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def request_llm(prompt, temperature=0.0, max_tokens=1200):
    """调用 OpenAI API，带重试"""
    for attempt in range(5):
        try:
            params = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "n": 1,
            }
            try:
                params["max_tokens"] = max_tokens
                resp = client.chat.completions.create(**params)
            except (TypeError, openai.BadRequestError):
                params.pop("max_tokens", None)
                params["max_completion_tokens"] = max_tokens * 2
                resp = client.chat.completions.create(**params)

            return resp.choices[0].message.content.strip()

        except openai.RateLimitError:
            wait = 2 ** attempt
            print(f"    速率限制，等待 {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"    LLM 错误 (attempt {attempt + 1}): {e}")
            time.sleep(1)
    return None


# ========================================================================== #
#  三元组提取
# ========================================================================== #

def parse_triples_response(response: str):
    """解析 LLM 返回的三元组文本 → 结构化字典列表

    支持两种格式：
      Triple: (head, relation, tail)
      Triple: ("head", "relation", "tail")
    """
    if not response:
        return []

    # 找到第一个 "Triple" 出现的位置
    start = response.find("Triple")
    if start == -1:
        return []
    response = response[start:]

    triples = []
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 提取 ": " 后面的内容
        if ": " in line:
            content = line.split(": ", 1)[1].strip()
        elif ":" in line:
            content = line.split(":", 1)[1].strip()
        else:
            content = line

        # 去掉外层括号
        content = content.strip()
        if content.startswith("(") and content.endswith(")"):
            content = content[1:-1]

        # 按顶层逗号分割三部分
        parts = []
        current = []
        depth = 0
        for ch in content:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip().strip('"\''))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip().strip('"\''))

        if len(parts) == 3 and all(p.strip() for p in parts):
            triples.append({
                "head":     parts[0],
                "relation": parts[1],
                "tail":     parts[2],
            })

    return triples


def extract_triples_two_stage(text: str):
    """两阶段三元组提取：初始提取 + 验证修正"""
    if not text or not text.strip():
        return []

    # 阶段 1：初始提取
    prompt1 = GPT_TRIPLE_EXTRACTION_PROMPT.format(init_text=text)
    response1 = request_llm(prompt1, temperature=0.0)
    if not response1:
        return []

    # 阶段 2：验证修正
    prompt2 = GPT_TRIPLE_EXTRACTION_PROMPT_REVISE.format(p=prompt1, t=response1)
    response2 = request_llm(prompt2, temperature=0.0)

    triples = parse_triples_response(response2 or response1)
    return triples


# ========================================================================== #
#  主处理流程
# ========================================================================== #

def process_mintaka_triples(input_path, output_path):
    print(f"加载数据: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    total = len(data)

    # 断点续传
    processed_data = load_progress(output_path)
    start_index = len(processed_data)
    if start_index > 0:
        print(f"已有进度: {start_index} 条，从第 {start_index + 1} 条继续...")
    print(f"总数据量: {total}\n")

    start_time = time.time()

    try:
        for idx in range(start_index, total):
            item = data[idx].copy()
            item_id  = item.get("id", str(idx))
            question = item.get("question", "")
            label    = item.get("generation_label", "")
            gpt_text = item.get("gpt_sentence", "")

            print(f"{'=' * 80}")
            print(f"[{len(processed_data) + 1}/{total}]  ID: {item_id}  标签: {label}")
            print(f"问题: {question}")

            try:
                if gpt_text and gpt_text.strip():
                    print(f"  提取 gpt_sentence 三元组（两阶段）...")
                    triples = extract_triples_two_stage(gpt_text)
                    print(f"  → 提取到 {len(triples)} 个三元组")
                else:
                    triples = []
                    print(f"  ⚠️  gpt_sentence 为空，跳过")

                item["gpt_triples"]      = triples
                item["gpt_triple_count"] = len(triples)
                processed_data.append(item)

                # 每 5 条保存一次
                if len(processed_data) % 5 == 0:
                    elapsed = time.time() - start_time
                    done    = len(processed_data) - start_index
                    avg     = elapsed / max(done, 1)
                    remaining = (total - len(processed_data)) * avg
                    print(f"\n💾 保存进度 ({len(processed_data)}/{total})  "
                          f"均速 {avg:.1f}s/条  预计剩余 {remaining / 60:.1f} 分钟")
                    save_data(processed_data, output_path)

                time.sleep(0.3)

            except Exception as e:
                print(f"  ❌ 处理出错: {e}")
                import traceback
                traceback.print_exc()
                save_data(processed_data, output_path)
                print("  5 秒后继续...")
                time.sleep(5)
                continue

    except KeyboardInterrupt:
        save_data(processed_data, output_path)
        print(f"\n⏸️  用户中断，进度已保存 ({len(processed_data)}/{total})")
        raise

    save_data(processed_data, output_path)
    total_time = time.time() - start_time

    total_triples = sum(len(it.get("gpt_triples", [])) for it in processed_data)

    print(f"\n{'=' * 80}")
    print(f"处理完成！结果已保存到: {output_path}")
    print(f"\n📊 统计信息:")
    print(f"  - 总数据量:          {len(processed_data)}")
    print(f"  - GPT 三元组总数:    {total_triples}")
    print(f"  - 平均三元组/条:     {total_triples / max(len(processed_data), 1):.1f}")
    print(f"  - 总耗时:            {total_time / 60:.1f} 分钟")
    print(f"  - 平均耗时:          {total_time / max(len(processed_data) - start_index, 1):.2f} 秒/条")

    # 标签分布
    correct_cnt      = sum(1 for it in processed_data if it.get("generation_label") == "correct")
    hallucination_cnt = sum(1 for it in processed_data if it.get("generation_label") == "hallucination")
    print(f"\n  标签分布:")
    print(f"    correct:      {correct_cnt}")
    print(f"    hallucination:{hallucination_cnt}")

    return processed_data


# ========================================================================== #
#  入口
# ========================================================================== #

if __name__ == "__main__":
    input_file  = "data/mintaka_dev_with_wikidata_triples.json"
    output_file = "data/mintaka_dev_with_gpt_triples.json"

    os.makedirs("data", exist_ok=True)

    print(f"""
{'=' * 80}
Mintaka GPT 推理文本三元组提取脚本
{'=' * 80}
配置:
  输入: {input_file}
  输出: {output_file}
  模型: {LLM_MODEL}

流程:
  阶段 1  初始提取 — 从 gpt_sentence 提取所有知识三元组
  阶段 2  验证修正 — LLM 检查并修正代词/语义错误
  输出字段: gpt_triples（结构化字典列表，格式与 entity_triples 一致）

断点续传: 每处理 5 条自动保存，中断后重新运行自动续传。
{'=' * 80}
    """)

    try:
        result = process_mintaka_triples(input_file, output_file)
        print(f"\n✅ 成功处理 {len(result)} 条数据")
    except KeyboardInterrupt:
        print("\n⏸️  用户中断，进度已保存")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
