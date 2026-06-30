import os
import json
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from queue import Queue
from threading import Lock
# INSERT_YOUR_CODE
import requests

import dotenv
import argparse
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

SENSITIVE_CHECK_URL = "https://spam.dw-dengwei.workers.dev"
SENSITIVE_CHECK_MAX_RETRIES = 3
SENSITIVE_CHECK_TIMEOUT_SECONDS = 5
SENSITIVE_CHECK_RETRY_DELAY_SECONDS = 1

DEFAULT_AI_FIELDS = {
    "tldr": "Summary generation failed",
    "motivation": "Motivation analysis unavailable",
    "method": "Method extraction failed",
    "result": "Result analysis unavailable",
    "conclusion": "Conclusion extraction failed"
}
REQUIRED_AI_FIELDS = tuple(DEFAULT_AI_FIELDS.keys())
JSON_OUTPUT_INSTRUCTION = """
Return only one valid JSON object. Do not use markdown code fences or any extra text.
The JSON object must contain exactly these string keys:
"tldr", "motivation", "method", "result", "conclusion".
Write all values in {language}.
If content must be hidden for compliance reasons, use the required hidden-content message as every JSON value.
"""

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()

def is_sensitive(content: str) -> bool:
    """
    调用 spam.dw-dengwei.workers.dev 接口检测内容是否包含敏感词。
    只有接口正常返回 sensitive=true 时才过滤内容；接口不可用时跳过检查继续处理。
    """
    last_error = None
    for attempt in range(1, SENSITIVE_CHECK_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                SENSITIVE_CHECK_URL,
                json={"text": content},
                timeout=SENSITIVE_CHECK_TIMEOUT_SECONDS
            )
            if resp.status_code == 200:
                try:
                    result = resp.json()
                except ValueError as e:
                    last_error = f"invalid JSON response: {e}"
                    print(
                        f"Sensitive check failed on attempt {attempt}/{SENSITIVE_CHECK_MAX_RETRIES}: {last_error}",
                        file=sys.stderr
                    )
                else:
                    # 约定接口返回 {"sensitive": true/false, ...}
                    return result.get("sensitive") is True
            else:
                last_error = f"status {resp.status_code}"
                print(
                    f"Sensitive check failed on attempt {attempt}/{SENSITIVE_CHECK_MAX_RETRIES} with status {resp.status_code}",
                    file=sys.stderr
                )
        except Exception as e:
            last_error = str(e)
            print(
                f"Sensitive check error on attempt {attempt}/{SENSITIVE_CHECK_MAX_RETRIES}: {e}",
                file=sys.stderr
            )

        if attempt < SENSITIVE_CHECK_MAX_RETRIES:
            time.sleep(SENSITIVE_CHECK_RETRY_DELAY_SECONDS)

    print(
        f"WARNING: Sensitive check unavailable after {SENSITIVE_CHECK_MAX_RETRIES} attempts; "
        f"treating content as not sensitive. Last error: {last_error}",
        file=sys.stderr
    )
    return False

def get_response_content(response) -> str:
    """Return plain text from a LangChain response object or string."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "\n".join(part for part in parts if part)
    return str(content)

def iter_json_candidates(text: str):
    """Yield likely JSON object strings from model output."""
    text = text.strip()
    fence_pattern = r"```(?:json)?\s*(.*?)```"
    for match in re.finditer(fence_pattern, text, flags=re.IGNORECASE | re.DOTALL):
        candidate = match.group(1).strip()
        if candidate:
            yield candidate

    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[start:idx + 1]
                        break
        start = text.find("{", start + 1)

def parse_ai_json_response(response, item_id: str) -> Dict:
    """Parse the model's plain JSON response and normalize AI fields."""
    content = get_response_content(response)
    last_error = None
    for candidate in iter_json_candidates(content):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e
            continue
        if not isinstance(parsed, dict):
            last_error = ValueError("JSON response is not an object")
            continue

        normalized = {
            field: str(parsed.get(field, DEFAULT_AI_FIELDS[field]))
            for field in REQUIRED_AI_FIELDS
        }
        try:
            if hasattr(Structure, "model_validate"):
                structured = Structure.model_validate(normalized)
            else:
                structured = Structure(**normalized)
            if hasattr(structured, "model_dump"):
                return structured.model_dump()
            return structured.dict()
        except Exception as e:
            last_error = e
            continue

    preview = content.replace("\n", " ")[:500]
    raise ValueError(f"Failed to parse JSON AI response for {item_id}: {last_error}; response preview: {preview}")

def process_single_item(chain, item: Dict, language: str) -> Dict:
    def check_github_code(content: str) -> Dict:
        """提取并验证 GitHub 链接"""
        code_info = {}

        # 1. 优先匹配 github.com/owner/repo 格式
        github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
        match = re.search(github_pattern, content)
        
        if match:
            owner, repo = match.groups()
            # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
            repo = repo.rstrip(".git").rstrip(".,)")
            
            full_url = f"https://github.com/{owner}/{repo}"
            code_info["code_url"] = full_url
            
            # 尝试调用 GitHub API 获取信息
            github_token = os.environ.get("TOKEN_GITHUB")
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            try:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                resp = requests.get(api_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    code_info["code_stars"] = data.get("stargazers_count", 0)
                    code_info["code_last_update"] = data.get("pushed_at", "")[:10]
            except Exception:
                # API 调用失败不影响主流程
                pass
            return code_info

        # 2. 如果没有 github.com，尝试匹配 github.io
        github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
        match_io = re.search(github_io_pattern, content)
        
        if match_io:
            url = match_io.group(0)
            # 清理末尾标点
            url = url.rstrip(".,)")
            code_info["code_url"] = url
            # github.io 不进行 star 和 update 判断
                
        return code_info

    # 检查 summary 字段
    if is_sensitive(item.get("summary", "")):
        return None

    # 检测代码可用性
    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    """处理单个数据项"""
    try:
        response = chain.invoke({
            "language": language,
            "content": item['summary']
        })
        item['AI'] = parse_ai_json_response(response, item.get('id', 'unknown'))
    except Exception as e:
        # Catch any other exceptions and provide default values
        print(f"Unexpected error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item['AI'] = DEFAULT_AI_FIELDS.copy()
    
    # Final validation to ensure all required fields exist
    for field in DEFAULT_AI_FIELDS.keys():
        if field not in item['AI']:
            item['AI'][field] = DEFAULT_AI_FIELDS[field]

    # 检查 AI 生成的所有字段
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return None
    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    llm = ChatOpenAI(model=model_name)
    print('Connect to:', model_name, file=sys.stderr)
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system + "\n" + JSON_OUTPUT_INSTRUCTION),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm
    
    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language): idx
            for idx, item in enumerate(data)
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # Add default AI fields to ensure consistency
                processed_data[idx] = data[idx]
                processed_data[idx]['AI'] = {
                    "tldr": "Processing failed",
                    "motivation": "Processing failed",
                    "method": "Processing failed",
                    "result": "Processing failed",
                    "conclusion": "Processing failed"
                }
    
    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)
    
    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers
    )
    
    # 保存结果
    written_count = 0
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")
                written_count += 1

    print(f"AI enhanced output count: {written_count}/{len(data)}", file=sys.stderr)
    if data and written_count == 0:
        print(
            f"ERROR: AI enhancement produced an empty output file from {len(data)} input papers: {target_file}",
            file=sys.stderr
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
