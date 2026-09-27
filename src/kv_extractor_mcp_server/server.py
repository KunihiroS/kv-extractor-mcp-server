from fastmcp import FastMCP
from pydantic_ai import Agent
from typing import Any, Dict, List, Union
from pydantic import BaseModel
import yaml
import json
import toml
import logging
import traceback
import os
import spacy
from spacy.util import is_package
from langdetect import detect
import sys
import argparse
import importlib
import importlib.util
import shutil
import subprocess

# MCP stdio transport: stdout carries JSON-RPC only. Every diagnostic line in this
# module goes to stderr (or the log file); never print() to stdout.

# --- Logging Setup ---
logger = logging.getLogger("kv-extractor-mcp-server")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.propagate = False

def setup_logging(log: str, logfile: str):
    if log == "on":
        if not logfile or not logfile.startswith("/"):
            print("[FATAL] --logfile must be an absolute path when --log=on", file=sys.stderr)
            sys.exit(1)
        log_dir = os.path.dirname(logfile)
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e:
                print(f"[FATAL] Failed to create log directory: {log_dir} error={e}", file=sys.stderr)
                sys.exit(1)
        try:
            # --- Full handler initialization ---
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            root_logger.handlers.clear()
            root_logger.setLevel(logging.DEBUG)

            # Add only FileHandler to root logger
            file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            # Also add StreamHandler (stderr; stdout is reserved for JSON-RPC)
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setLevel(logging.DEBUG)
            stream_handler.setFormatter(formatter)
            root_logger.addHandler(stream_handler)

            # Explicitly call flush
            file_handler.flush()

            # Set global logger as well
            specific_logger = logging.getLogger("kv-extractor-mcp-server")
            specific_logger.handlers.clear()  # Clear existing handlers
            specific_logger.setLevel(logging.DEBUG)
            specific_logger.addHandler(file_handler)
            specific_logger.addHandler(stream_handler)
            specific_logger.propagate = False

            specific_logger.debug(f"=== MCP Server log initialized: {logfile} ===")
            print(f"[INFO] MCP Server log initialized: {logfile}", file=sys.stderr)
            if os.path.exists(logfile):
                print(f"[INFO] Log file created: {logfile}", file=sys.stderr)
            else:
                print(f"[WARN] Log file NOT created: {logfile}", file=sys.stderr)

            # Return the logger
            return specific_logger
        except Exception as e:
            print(f"[FATAL] Failed to create log file: {logfile} error={e}", file=sys.stderr)
            sys.exit(1)
    elif log == "off":
        # Completely disable logging functionality
        logging.disable(logging.CRITICAL)
        print("[INFO] Logging disabled (--log=off)", file=sys.stderr)
        return None
    else:
        print("[FATAL] --log must be 'on' or 'off'", file=sys.stderr)
        sys.exit(1)

# --- CLI Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Flexible Key-Value Extraction MCP Server")
    parser.add_argument("--log", choices=["on", "off"], required=True, help="Enable or disable logging (on/off)")
    parser.add_argument("--logfile", type=str, required=False, help="Absolute path to log file (required if --log=on)")
    args = parser.parse_args()
    if args.log == "on" and not args.logfile:
        print("[FATAL] --logfile is required when --log=on", file=sys.stderr)
        sys.exit(1)
    return args

# --- LLM Agents ---
# Main processing agent (lightweight model) and evaluation agent (high-precision model).
# They are created lazily on the first tool call so that importing this module,
# `--help`, and MCP initialize / tools/list all work without OPENAI_API_KEY.
MAIN_MODEL = 'openai:gpt-4.1-mini'
EVAL_MODEL = 'openai:gpt-4.1'
_agents: Dict[str, Agent] = {}


class MissingAPIKeyError(RuntimeError):
    """Raised when a tool is called but no OpenAI API key is configured."""


def get_agents():
    """Return (agent_main, agent_eval), creating them on first use."""
    if not _agents:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is not set. Provide it in the MCP host configuration (env) for this server."
            )
        # pydantic-ai prints a banner to stderr on first run; keep the stream quiet.
        os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
        _agents["main"] = Agent(MAIN_MODEL)
        _agents["eval"] = Agent(EVAL_MODEL)
    return _agents["main"], _agents["eval"]

# --- Multilingual Support: spaCy NER Preprocessing (Step 0) ---
LANG_MODEL_MAP = {
    'ja': 'ja_core_news_md',
    'en': 'en_core_web_sm',
    'zh-cn': 'zh_core_web_sm',
    'zh-tw': 'zh_core_web_sm',
}

def _spacy_model_install_command(model_name: str) -> List[str]:
    """Build the installer command for a spaCy pipeline package.

    `spacy.cli.download` is not used because (a) when the interpreter has no pip
    (e.g. a `uvx` environment) it falls back to a bare `uv pip install`, which
    targets whatever virtualenv `uv` discovers from the MCP host's working
    directory instead of this interpreter, and (b) its failures call sys.exit(),
    which would terminate the MCP server from inside a tool call.
    """
    from spacy import about
    from spacy.cli.download import get_compatibility, get_model_filename, get_version

    try:
        version = get_version(model_name, get_compatibility())
    except SystemExit as e:  # spaCy reports resolution failures via sys.exit()
        raise RuntimeError(
            f"Could not resolve a spaCy model compatible with spaCy {spacy.__version__} for '{model_name}'"
        ) from e
    url = about.__download_url__.rstrip("/") + "/" + get_model_filename(model_name, version)
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", url]
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, url]
    raise RuntimeError(
        f"Cannot install spaCy model '{model_name}': neither pip nor uv is available. "
        f"Install it manually into {sys.executable} from {url}"
    )


def _install_spacy_model(model_name: str) -> None:
    """Install a spaCy pipeline package into this interpreter's environment.

    Installer output is captured and forwarded to stderr: on the MCP stdio
    transport stdout must carry JSON-RPC only.
    """
    cmd = _spacy_model_install_command(model_name)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if proc.stdout:
        sys.stderr.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    sys.stderr.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"Installing spaCy model '{model_name}' failed (exit {proc.returncode})")
    # The package was added to site-packages after startup; drop the import
    # system's cached directory listings so spacy.load() can find it.
    importlib.invalidate_caches()


def get_spacy_model_for_text(text: str):
    # 1. Language detection
    try:
        lang = detect(text)
    except Exception:
        raise ValueError("Language detection failed")
    if lang not in LANG_MODEL_MAP:
        raise ValueError(f"Unsupported lang detected: {lang}")
    model_name = LANG_MODEL_MAP[lang]
    # 2. Check and install model if necessary
    if not is_package(model_name):
        _install_spacy_model(model_name)
    # 3. Load model
    return spacy.load(model_name), lang

def extract_phrases_with_spacy_multilang(text: str):
    try:
        nlp, lang = get_spacy_model_for_text(text)
    except ValueError as e:
        # Error response example
        return {"success": False, "error": str(e)}
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

# --- Few-shot Examples by Language ---
FEW_SHOT_KV_EXTRACTION = {
    'ja': [
        {"input": "田中さんと佐藤さんが資料作成を担当します。", "output": "key: person, value: [田中, 佐藤]\nkey: task, value: 資料作成"},
        {"input": "明日は4月25日です。天気は晴れ、最高気温は25℃の予想です。", "output": "key: date, value: 20250425\nkey: weather, value: clear\nkey: temperature, value: 25"},
    ],
    'en': [
        {"input": "John and Mike will attend a meeting in Tokyo.", "output": "key: person, value: [John, Mike]\nkey: location, value: Tokyo"},
        {"input": "Tomorrow is April 25th. The weather is clear, with a high of 25°C expected.", "output": "key: date, value: 20250425\nkey: weather, value: clear\nkey: temperature, value: 25"},
    ],
    'zh-cn': [
        {"input": "张三和李四负责准备材料。", "output": "key: person, value: [张三, 李四]\nkey: task, value: 准备材料"},
        {"input": "明天是4月25日。天气晴朗，预计最高气温为25℃。", "output": "key: date, value: 20250425\nkey: weather, value: 晴朗\nkey: temperature, value: 25"},
    ],
    'zh-tw': [
        {"input": "張三和李四負責準備材料。", "output": "key: person, value: [張三, 李四]\nkey: task, value: 準備材料"},
        {"input": "明天是4月25日。天氣晴朗，預計最高氣溫為25℃。", "output": "key: date, value: 20250425\nkey: weather, value: 晴朗\nkey: temperature, value: 25"},
    ],
}

# ------ Utility: Get few-shot examples for language ------
def get_few_shot_examples(lang: str):
    return FEW_SHOT_KV_EXTRACTION.get(lang, FEW_SHOT_KV_EXTRACTION['en'])

# ------ Few-shot Examples (Primitive Style) ------
FEW_SHOT_TYPE_ANNOTATION = [
    {
        "input": """key: person, value: John -> str
key: date, value: 20240501 -> int
key: location, value: Tokyo -> str
key: event, value: meeting -> str
key: start_time, value: 10:00 -> str""",
        "output": """key: person, value: John -> str
key: date, value: 20240501 -> int
key: location, value: Tokyo -> str
key: event, value: meeting -> str
key: start_time, value: 10:00 -> str"""
    },
    {
        "input": """key: date, value: 20250425 -> int
key: weather, value: clear -> str
key: temperature, value: 25 -> int""",
        "output": """key: date, value: 20250425 -> int
key: weather, value: clear -> str
key: temperature, value: 25 -> int"""
    },
    {
        "input": """key: order_id, value: 12345 -> int
key: total_amount, value: 12800 -> int
key: shipping_date, value: 20250425 -> int""",
        "output": """key: order_id, value: 12345 -> int
key: total_amount, value: 12800 -> int
key: shipping_date, value: 20250425 -> int"""
    },
    {
        "input": """key: phone_number, value: 03-1234-5678 -> str
key: support_email, value: support@example.com -> str""",
        "output": """key: phone_number, value: 03-1234-5678 -> str
key: support_email, value: support@example.com -> str"""
    },
    {
        "input": """key: employee_id, value: A00123 -> str
key: registration_date, value: 20241201 -> int""",
        "output": """key: employee_id, value: A00123 -> str
key: registration_date, value: 20241201 -> int"""
    },
]

LLM_EXTRACTION_SYSTEM_PROMPT = """
You are an AI that extracts key-value information from given text.
【**Strict Rules**】
- **Do not output any information not present in the original text.**
- **Both keys and values must be explicitly present in the original text.**
- **Do not include any unnecessary keys or template-like information.**
- **Only extract keys that are explicitly present in the original text.**
- **Output must strictly follow the specified format (JSON/YAML/TOML).**
"""

def build_kv_extraction_prompt(input_text: str, spacy_phrases: list[str] = None, lang: str = 'en'):
    language_map = {
        'ja': 'Japanese',
        'en': 'English',
        'zh-cn': 'Chinese (Simplified)',
        'zh-tw': 'Chinese (Traditional)',
    }
    lang_label = language_map.get(lang, 'English')
    prompt = (
        f"## Language context: {lang_label}\n"
        "Please output primarily in the detected input language, "
        "but do not forcibly translate technical terms, proper nouns, or common English expressions. "
        "Retain the original wording for software, business, and technical contexts as appropriate.\n"
        "Extract key-value pairs from the following input. For each important fact or entity, output as:\n"
        "key: <key>, value: <value>\n"
        "Instructions:\n"
        "- Extract information about all entities (people, events, tasks, decisions, etc.) without omission, even if there are multiple entities in the text.\n"
        "- If the same key (e.g., person, task) appears multiple times, output the value as a JSON array (e.g., key: person, value: [\"John\", \"Mike\", \"Lisa\"]).\n"
        "- If the value is a complex object (e.g., a person with role and name), output the value as a JSON object or array of objects.\n"
        "- For bulleted or list-formatted text, evaluate each item individually and extract related key-value pairs.\n"
        "- Split the text into sentences or paragraphs if necessary, and evaluate each part separately to ensure no information is missed.\n"
        "- Avoid omitting any information; include all relevant details even if they seem minor.\n"
        "- Output must strictly follow: key: <key>, value: <value> per line.\n"
        "- All lists and objects must be valid JSON. Do NOT use YAML or TOML.\n"
        "- Do NOT use any custom or pseudo-list notation (like [A, B, C] without quotes).\n"
        "- Do NOT use YAML structure.\n"
        "Examples:\n"
    )
    for ex in get_few_shot_examples(lang):
        prompt += f"Input: {ex['input']}\n{ex['output']}\n"
    prompt += f"Input: {input_text}\n"
    if spacy_phrases:
        prompt += (
            "\n[Preprocessing Candidate Phrases (spaCy NER)]\n"
            "The following is a list of phrases automatically extracted from the input text using spaCy's Named Entity Recognition (NER) model.\n"
            "These phrases represent detected entities such as names, dates, organizations, locations, numbers, etc.\n"
            f"{spacy_phrases}\n"
            "This list is for reference only and may contain irrelevant or incorrect items. You should use your own judgment and consider the entire input text to flexibly infer the most appropriate key-value pairs.\n"
        )
    prompt += LLM_EXTRACTION_SYSTEM_PROMPT
    return prompt

def build_entity_extraction_prompt(input_text: str) -> str:
    prompt = (
        "List all significant entities (people, events, tasks, decisions, etc.) from the following input. Output as:\n"
        "entity: <entity>\n"
        "Do not use any JSON or YAML structure. List one per line.\n"
        "Instructions:\n"
        "- Identify all entities without omission, even if there are multiple entities in the text.\n"
        "- Split the text into sentences or paragraphs if necessary, and evaluate each part separately to ensure no entity is missed.\n"
        "Examples:\n"
        "Input: 田中さんは資料作成、佐藤さんは会議準備をお願いします。\n"
        "entity: 田中\n"
        "entity: 佐藤\n"
        "Input: {input_text}\n"
    )
    return prompt

def build_entity_detail_extraction_prompt(entity: str, input_text: str) -> str:
    prompt = (
        f"Extract details related to the entity '{entity}' from the following input. Output as:\n"
        "key: <key>, value: <value>\n"
        "Do not use any JSON or YAML structure. List one per line.\n"
        "Instructions:\n"
        "- Extract all relevant details (tasks, times, locations, etc.) associated with the entity.\n"
        "- Ensure no related information is omitted.\n"
        "Examples:\n"
        "Input: 田中さんは資料作成、佐藤さんは会議準備をお願いします。 Entity: 田中\n"
        "key: person, value: 田中\n"
        "key: task, value: 資料作成\n"
        "Input: {input_text}\n"
        "Entity: {entity}\n"
    )
    return prompt

def build_type_annotation_prompt(kv_lines: str) -> str:
    prompt = (
        "For each key-value pair below, infer the type of the value (e.g. int, str, date, time, float, bool, etc.).\n"
        "Rules:\n"
        "- Pure numbers (e.g. 123, 45.6) should be int or float, not str.\n"
        "- Dates (e.g. 20250425, 2024/12/01) should be int or str depending on context, but prefer int for YYYYMMDD.\n"
        "- Phone numbers, IDs, postal codes, etc. should be str, even if numeric.\n"
        "- Amounts or prices (e.g. 12800, 89,800) should be int or float.\n"
        "- If unsure, choose the most natural type for the value.\n"
        "Examples:\n"
    )
    for ex in FEW_SHOT_TYPE_ANNOTATION:
        prompt += f"Input: {ex['input']}\n{ex['output']}\n"
    prompt += f"{kv_lines}\n"
    return prompt

def build_evaluation_prompt(typed_lines: str) -> str:
    return (
        "You are an expert type validator. Review the following key-value-type list:\n"
        f"{typed_lines}\n\n"
        "Return the complete list with corrected types. Each line must follow the format 'key: value -> type' exactly as shown in the input.\n"
        "If all types are already correct, simply return the original list unchanged.\n\n"
        "Examples:\n"
        "Input:\n"
        "key: date, value: 2024/05/07 -> str\n"
        "key: count, value: 5 -> str\n\n"
        "Output (corrected):\n"
        "key: date, value: 2024/05/07 -> str\n"
        "key: count, value: 5 -> int\n"
    )

# ------ Pydantic Models ------
class KVOut(BaseModel):
    key: str
    value: Union[Any, List[Any], None]
    type: str

class KVPayload(BaseModel):
    kvs: List[KVOut]

# ------ Type Normalization (Step 4) ------
import re
import asyncio

# --- Static Normalization Rules ---
# Order matters: More specific patterns should come first.
STATIC_NORMALIZATION_RULES = [
    # Null/Empty values
    {
        'name': 'null_check',
        'pattern': r'^(N/A|ー|-|null|undefined|none|)$', # Added empty string
        'handler': lambda _, __: None, # Always return None
        'target_types': ['str', 'int', 'float', 'bool', 'date', 'any'] # Apply to almost any expected type
    },
    # Boolean values
    {
        'name': 'boolean',
        'pattern': r'^(true|false|yes|no|有効|無効|はい|いいえ|1|0|あり|なし)$', # Added 'あり', 'なし'
        'handler': lambda v, __: v.lower() in ('true', 'yes', '有効', 'はい', '1', 'あり'), # Added 'あり'
        'target_types': ['bool', 'str'] # Added 'str' to apply even if type is string
    },
    # Currency/Numbers (int/float) - Remove symbols and commas
    {
        'name': 'currency_int',
        'pattern': r'^[¥$€￥]?[\d,]+(円|元|人民币|CNY|TWD|新台币)?$',
        'handler': lambda v, __: int(re.sub(r'[¥$€￥,円元人民币CNYTWD新台币]', '', v)),
        'target_types': ['int', 'integer']
    },
    {
        'name': 'currency_float',
        'pattern': r'^[¥$€￥]?[\d,]+\.\d+(円|元|人民币|CNY|TWD|新台币)?$',
        'handler': lambda v, __: float(re.sub(r'[¥$€￥,円元人民币CNYTWD新台币]', '', v)),
        'target_types': ['float']
    },
    # Basic Integers
    {
        'name': 'basic_int',
        'pattern': r'^[+-]?\d+$',
        'handler': lambda v, __: int(v),
        'target_types': ['int', 'integer']
    },
    # Basic Floats
    {
        'name': 'basic_float',
        'pattern': r'^[+-]?\d+\.\d+$',
        'handler': lambda v, __: float(v),
        'target_types': ['float']
    },
    # Date Formats (YYYY/MM/DD, YYYY-MM-DD, YYYYMMDD) -> YYYYMMDD string
    {
        'name': 'date_ymd',
        'pattern': r'^(\d{4})[/-]?(\d{1,2})[/-]?(\d{1,2})$',
        'handler': lambda v, m: f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}",
        'target_types': ['date', 'str'] # Can be string or specific date type
    },
    # Phone Numbers (remove hyphens) -> string
    {
        'name': 'phone',
        'pattern': r'^\d{2,4}-?\d{2,4}-?\d{4}$',
        'handler': lambda v, __: re.sub(r'\D', '', v),
        'target_types': ['str']
    },
    # Comma-separated list -> List[str]
    {
        'name': 'comma_list',
        # Improved pattern: Matches strings with at least one comma separating non-comma characters.
        # Handles potential spaces around commas.
        'pattern': r'^[^,]+(\s*,\s*[^,]+)+$',
        'handler': lambda v, __: [s.strip() for s in re.split(r'\s*,\s*', v)], # Split considering spaces
        'target_types': ['list', 'array', 'str'] # Added 'str' to apply even if type is string
    }
]

# --- LLM Normalization Helper ---
async def llm_normalize_value(key: str, value: str, expected_type: str, agent: Agent) -> Any:
    """Uses LLM to normalize a value when static rules fail."""
    prompt = f"""\
Given the key-value pair and the expected type, normalize the value into the most appropriate format.\n\nKey: {key}\nValue: \"{value}\"\nExpected Type: {expected_type}\n\nNormalization Rules:\n- Dates: Convert to YYYYMMDD string format (e.g., \"May 15th, 2024\" -> \"20240515\").\n- Booleans: Convert natural language (yes/no, 有効/無効) to true/false.\n- Numbers: Remove currency symbols, commas, units (e.g., \"¥12,800\" -> 12800, \"25℃\" -> 25). Return int or float.\n- Lists: Always return a valid JSON array. If the value is a comma-separated string, enumerate, or list-formatted, convert to a JSON array of strings.\n- Objects: Always return a valid JSON object or array of objects if appropriate.\n- Nulls: Convert \"N/A\", \"-\", \"none\" to null.\n- Other strings: Clean up extra whitespace. Phone numbers should be digits only.\n\nReturn ONLY the normalized value, as a valid JSON value if it is a list or object. Do not return any extra explanation or formatting. If normalization is not possible or doesn't make sense, return the original value.\nExamples:\nValue: \"May 1st\", Expected Type: date -> 20240501 (assuming current year if not specified)\nValue: \"today\", Expected Type: date -> 20250426 (current date)\nValue: \"有効\", Expected Type: bool -> true\nValue: \"12,800円\", Expected Type: int -> 12800\nValue: \"apple, banana\", Expected Type: list -> [\"apple\", \"banana\"]\nValue: \"Colors are Red, Green, Blue\", Expected Type: list -> [\"Red\", \"Green\", \"Blue\"]\nValue: \"N/A\", Expected Type: str -> null\nValue: \"Tokyo Tower\", Expected Type: str -> Tokyo Tower\n\nNormalized Value:"""
    try:
        logging.debug(f"Attempting LLM normalization for key='{key}', value='{value}', type='{expected_type}'")
        result = await agent.run(prompt)
        normalized_str = result.output.strip()
        logging.debug(f"LLM normalization raw result: '{normalized_str}'")

        # Attempt to parse the LLM output based on expected type
        if normalized_str.lower() == 'null':
            return None
        if expected_type == 'bool':
            return normalized_str.lower() == 'true'
        if expected_type == 'int':
            return int(normalized_str)
        if expected_type == 'float':
            return float(normalized_str)
        if expected_type in ['list', 'array', 'object', 'dict']:
            try:
                parsed = json.loads(normalized_str)
                return parsed
            except json.JSONDecodeError:
                logging.warning(f"LLM result for {expected_type} type wasn't valid JSON: {normalized_str}")
                return value
        # Default: return the string result from LLM (or original if LLM output seems invalid)
        return normalized_str if normalized_str else value

    except Exception as e:
        logging.error(f"Error during LLM normalization for key '{key}': {e}")
        return value # Fallback to original value on error

# --- Main Normalization Function ---
async def normalize_types_v2(kv_data: List[KVOut], agent_for_llm: Agent, lang: str) -> List[KVOut]:
    """
    Normalizes types using a hybrid approach: static rules first, then LLM fallback.
    
    Args:
        kv_data (List[KVOut]): List of key-value-type objects from previous steps.
        agent_for_llm (Agent): The Pydantic-AI agent to use for LLM-based normalization.
        lang (str): Detected language code (e.g., 'ja', 'en', 'zh-cn', 'zh-tw').
        
    Returns:
        List[KVOut]: Updated list with normalized values.
    """
    # Language-specific boolean normalization sets
    BOOL_TRUE_VALUES = {
        'ja': {'はい', '有効', '真', '1', 'ある'},
        'en': {'yes', 'true', 'valid', 'enabled', '1'},
        'zh-cn': {'是', '有', '对', '真', '对的', '有的', '1'},
        'zh-tw': {'是', '有', '對', '真', '對的', '有的', '1'},
    }
    BOOL_FALSE_VALUES = {
        'ja': {'いいえ', '無効', '偽', '0', 'ない'},
        'en': {'no', 'false', 'invalid', 'disabled', '0'},
        'zh-cn': {'否', '没有', '错', '假', '错误', '不对', '没有的', '0'},
        'zh-tw': {'否', '沒有', '錯', '假', '錯誤', '不對', '沒有的', '0'},
    }
    def lang_bool_handler(v, __):
        v_norm = v.lower().strip()
        if v_norm in BOOL_TRUE_VALUES.get(lang, set()):
            return True
        if v_norm in BOOL_FALSE_VALUES.get(lang, set()):
            return False
        return v  # fallback: return original

    # Make a local copy of STATIC_NORMALIZATION_RULES and patch boolean rule
    local_rules = [rule.copy() for rule in STATIC_NORMALIZATION_RULES]
    for rule in local_rules:
        if rule['name'] == 'boolean':
            rule['handler'] = lang_bool_handler
            # Optionally, update pattern as well for stricter matching
            rule['pattern'] = r'^({})$'.format(
                '|'.join(sorted(BOOL_TRUE_VALUES.get(lang, set()) | BOOL_FALSE_VALUES.get(lang, set())))
            )
    # ... (rest of normalize_types_v2 uses local_rules instead of STATIC_NORMALIZATION_RULES) ...
    normalized_data = []
    tasks = []

    async def process_item(item):
        key = str(item.key)
        value = item.value
        # Ensure value is string for pattern matching, unless it's already None
        value_str = str(value) if value is not None else ""
        expected_type = item.type.lower() if item.type else "str"
        normalized_value = value # Default to original
        processed_statically = False

        # List type detection and conversion enhancement (balanced)
        def is_list_like(val):
            if isinstance(val, list):
                return True
            if isinstance(val, str):
                s = val.strip()
                # Strict: Only treat as list if surrounded by []
                if s.startswith('[') and s.endswith(']'):
                    try:
                        import ast
                        parsed = ast.literal_eval(s)
                        return isinstance(parsed, list)
                    except Exception:
                        return False
            return False

        # Flexible and safe list parsing function (with debug log, fully flatten)
        def safe_parse_list(val):
            import ast
            import logging
            logging.debug(f"[safe_parse_list] input: {repr(val)} (type: {type(val)})")
            # If already a list, recursively flatten considering nested lists
            if isinstance(val, list):
                result = []
                for item in val:
                    parsed_item = None
                    if isinstance(item, str) and item.strip().startswith("[") and item.strip().endswith("]"):
                        try:
                            parsed = ast.literal_eval(item)
                            parsed_item = safe_parse_list(parsed)
                        except Exception:
                            parsed_item = item
                    else:
                        parsed_item = safe_parse_list(item) if isinstance(item, (list, str)) else item
                    if isinstance(parsed_item, list):
                        result.extend(parsed_item)
                    else:
                        result.append(parsed_item)
                # Fully flatten (expand any single-element list repeatedly)
                while len(result) == 1 and isinstance(result[0], list):
                    logging.debug(f"[safe_parse_list] flattening single-element list: {repr(result)}")
                    result = result[0]
                logging.debug(f"[safe_parse_list] output: {repr(result)} (type: {type(result)})")
                return result
            # List representation string surrounded by []
            if isinstance(val, str) and val.strip().startswith("[") and val.strip().endswith("]"):
                try:
                    parsed = ast.literal_eval(val)
                    logging.debug(f"[safe_parse_list] parsed list representation string: {repr(parsed)}")
                    return safe_parse_list(parsed)
                except Exception:
                    pass
            # Split by comma (only if it looks like a word list)
            if isinstance(val, str) and "," in val and not any(sep in val for sep in ['\n', '\r']):
                items = [v.strip() for v in val.split(",")]
                if len(items) >= 2 and all(len(x) < 30 for x in items):
                    logging.debug(f"[safe_parse_list] split by comma: {repr(items)}")
                    return items
            # Otherwise, return as is
            logging.debug(f"[safe_parse_list] returning as is: {repr(val)} (type: {type(val)})")
            return val

        # Always pass through safe_parse_list if type is list or str
        if isinstance(value, (list, str)):
            value = safe_parse_list(value)

        # 1. Try Static Rules
        for rule in local_rules:
            rule_name = rule['name']
            # --- Modified: Do not apply comma_list rule if already list type ---
            if rule_name == 'comma_list' and isinstance(value, list):
                continue
            if rule.get('target_types') and expected_type not in rule['target_types']:
                continue
            pattern = rule['pattern']
            handler = rule['handler']
            m = re.match(pattern, value_str)
            if m:
                logging.debug(f"Static rule '{rule_name}' applied to key '{key}'. Value '{value_str}' -> '{handler(value_str, m) if 'm' in handler.__code__.co_varnames else handler(value_str, expected_type)}")
                try:
                    # Some handlers need the match object, some just value and expected_type
                    if 'm' in handler.__code__.co_varnames:
                        normalized_value = handler(value_str, m)
                    else:
                        normalized_value = handler(value_str, expected_type)
                    processed_statically = True
                    break
                except Exception as e:
                    logging.error(f"Error in static rule '{rule_name}' for key '{key}': {e}")
                    continue

        # 2. LLM Fallback (if not processed statically and not None)
        if not processed_statically and value is not None:
            # Only use LLM if static rules didn't apply or failed, and value isn't already None
            logging.debug(f"Using LLM fallback for key '{key}', value '{value_str}', expected_type '{expected_type}'")
            normalized_value = await llm_normalize_value(key, value_str, expected_type, agent_for_llm)

        # Update the type based on the Python type of the normalized value
        final_type = type(normalized_value).__name__
        # Handle None separately
        if normalized_value is None:
            final_type = 'null' # Or keep original expected_type? Let's use 'null' for clarity.
        # Type information is also list type, so type="list"
        if isinstance(normalized_value, list):
            final_type = "list"

        return KVOut(key=key, value=normalized_value, type=final_type)

    # Process items concurrently
    results = await asyncio.gather(*(process_item(item) for item in kv_data))
    return results


# ------ Main Pipeline ------
async def extract_kv_pipeline(input_text: str, output_format: str) -> Dict[str, Any]:
    try:
        agent_main, agent_eval = get_agents()
    except MissingAPIKeyError as e:
        # Fail closed with an explicit, secret-free message instead of a traceback.
        return {"success": False, "error": str(e)}
    try:
        logging.debug(f"Starting extract_kv_pipeline with input_text: {input_text[:100]}... (truncated)")
        lang = detect(input_text)
        # Step 0: Preprocessing with spaCy (Named Entity Recognition)
        spacy_phrases = extract_phrases_with_spacy_multilang(input_text)
        # Step 1: Extract key-value pairs with LLM (pass preprocessing info as well)
        kv_prompt = build_kv_extraction_prompt(input_text, spacy_phrases, lang)
        logging.debug(f"KV extraction prompt: {kv_prompt[:200]}... (truncated)")
        kv_lines = await agent_main.run(kv_prompt)
        logging.debug(f"KV extraction result: {kv_lines.output[:200]}... (truncated)")

        # Step 2: Pass type annotation to LLM (as is)
        type_prompt = build_type_annotation_prompt(kv_lines.output)
        logging.debug(f"Type annotation prompt: {type_prompt[:200]}... (truncated)")
        typed_lines = await agent_main.run(type_prompt)
        logging.debug(f"Type annotation result: {typed_lines.output[:200]}... (truncated)")

        # Step 3: Specialized evaluation for type annotation
        eval_prompt = build_evaluation_prompt(typed_lines.output)
        logging.debug(f"Evaluation prompt: {eval_prompt[:200]}... (truncated)")
        eval_result = await agent_eval.run(eval_prompt)
        logging.debug(f"Evaluation result: {eval_result.output[:200]}... (truncated)")

        # Use the evaluation result directly - it's either the corrected or unchanged list
        # Add minimal format check as a safety measure
        if not any("->" in line for line in eval_result.output.splitlines()):
            # If result doesn't contain expected format, fall back to original
            logging.warning("Evaluation result doesn't contain expected format, using original")
            typed_lines_for_parse = typed_lines.output
        else:
            typed_lines_for_parse = eval_result.output
        logging.debug(f"Parsed lines for final step: {typed_lines_for_parse[:200]}... (truncated)")

        # Step 4: Formatting and type validation are fully delegated to pydantic-ai
        result = await agent_eval.run(
            f"Summarize the following key-value-type lines as a structured object with 'kvs' list. "
            f"Each kvs item should have key, value, and type. "
            f"Lines:\n{typed_lines_for_parse}",
            output_type=KVPayload
        )

        # Extract KVPayload from AgentRunResult
        if hasattr(result, "output") and isinstance(result.output, KVPayload):
            result = result.output
        elif isinstance(result, KVPayload):
            pass # Already correct type
        else:
            return {
                "success": False,
                "error": f"Unexpected result type: {type(result)}",
                "debug": {"result_repr": repr(result)}
            }

        # Add debug info to return value
        debug_info = {
            "type": str(type(result)),
            "dir": str(dir(result)),
            "repr": repr(result),
            "dict": str(getattr(result, "__dict__", repr(result)))
        }

        try:
            kv_data = result.kvs
            debug_info["kv_data_type"] = str(type(kv_data))
            debug_info["kv_data_repr"] = repr(kv_data)

            try:
                # Step 4 (New): Type normalization process (V2)
                # Pass agent_eval for LLM fallback
                normalized_kv_data = await normalize_types_v2(kv_data, agent_eval, lang)
                debug_info["normalized_kv_data_repr"] = repr(normalized_kv_data)

                # Step 5: Create final output dictionary using normalized data
                # The value is now already in the correct Python type
                output_dict = {}
                for item in normalized_kv_data:
                    k = str(item.key)
                    v = item.value
                    if k in output_dict:
                        # If existing value is a list, append; if single value, convert to list
                        if isinstance(output_dict[k], list):
                            if isinstance(v, list):
                                output_dict[k].extend(v)
                            else:
                                output_dict[k].append(v)
                        else:
                            output_dict[k] = [output_dict[k]]
                            if isinstance(v, list):
                                output_dict[k].extend(v)
                            else:
                                output_dict[k].append(v)
                    else:
                        # Explicitly guarantee the specification: set the value as is, without str()
                        # When storing values, do not convert types; keep list type as is
                        output_dict[k] = v  # Set value as is, without str()

                debug_info["output_dict_type"] = str(type(output_dict))
                debug_info["output_dict_repr"] = repr(output_dict)

                # Format output based on request (JSON is default dict, others need conversion)
                final_result = output_dict # Default JSON
                if output_format == "yaml":
                    final_result = yaml.dump(output_dict, allow_unicode=True)
                elif output_format == "toml":
                    # TOML requires careful handling, might need adjustments
                    # Basic conversion for now
                    try:
                        final_result = toml.dumps({"result": output_dict}) # Wrap in a table
                        # Remove the wrapper table header
                        final_result = final_result.replace("[result]\n", "", 1)
                    except Exception as toml_e:
                         logging.error(f"TOML conversion failed: {toml_e}")
                         # Fallback or return error? Let's return the dict with a warning.
                         debug_info["toml_conversion_error"] = str(toml_e)


                return {
                    "success": True,
                    "result": final_result
                }
            except Exception as dict_e:
                debug_info["normalization_or_output_exception"] = str(dict_e)
                debug_info["normalization_or_output_traceback"] = traceback.format_exc()
                return {
                    "success": False,
                    "error": "Exception in output_dict processing",
                    "debug": {k: str(v) for k, v in debug_info.items()}
                }
        except Exception as inner_e:
            debug_info["kv_data_exception"] = str(inner_e)
            debug_info["kv_data_traceback"] = traceback.format_exc()
            debug_info["result_repr"] = repr(result)
            return {
                "success": False,
                "error": "Exception in kv_data/output_dict processing",
                "debug": {k: str(v) for k, v in debug_info.items()}
            }

    except Exception as e:
        debug_info = {
            "exception": str(e),
            "traceback": traceback.format_exc()
        }
        return {"success": False, "error": str(e), "debug": debug_info}

# ---------- MCP Tools ----------

server = FastMCP(
    'Flexible Key-Value Extraction MCP Server'
)

@server.tool(
    name="extract_json",
    description="Extracts key-value pairs from arbitrary noisy text and returns them as type-safe JSON (dict)."
)
async def extract_json(input_text: str) -> dict:
    return await extract_kv_pipeline(input_text, "json")

@server.tool(
    name="extract_yaml",
    description="Extracts key-value pairs from arbitrary noisy text and returns them as type-safe YAML (str)."
)
async def extract_yaml(input_text: str) -> dict:
    return await extract_kv_pipeline(input_text, "yaml")

@server.tool(
    name="extract_toml",
    description="Extracts key-value pairs from arbitrary noisy text and returns them as type-safe TOML (str)."
)
async def extract_toml(input_text: str) -> dict:
    return await extract_kv_pipeline(input_text, "toml")

# Server initialization and run (MCP host support)
def initialize_and_run_server():
    try:
        args = parse_args()
        logger = setup_logging(args.log, args.logfile)
        if logger:
            logger.info("MCP Server starting up via entry point...")
        server.run(show_banner=False)
    except Exception as e:
        print(f"[FATAL] Error during server initialization: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logging(args.log, args.logfile)
    if logger:
        logger.info("MCP Server starting up...")
    server.run(show_banner=False)