#!/usr/bin/env python3
"""
scripts/generate_training_data.py

Uses Gemini, DeepSeek, Mistral, and/or LM Studio (local) as teachers to generate Korean training data.

Usage:
    python scripts/generate_training_data.py --teacher gemini --batches 500
    python scripts/generate_training_data.py --teacher deepseek --batches 500
    python scripts/generate_training_data.py --teacher mistral --batches 500
    python scripts/generate_training_data.py --teacher lmstudio --batches 500
    python scripts/generate_training_data.py --teacher all --batches 200  # 200*4 = 800 batches

LM Studio Setup:
    1. Download from https://lmstudio.ai
    2. Load a Korean-capable model (e.g., Qwen, Mistral, Llama 3)
    3. Start the local server (default: http://localhost:1234)
    4. Run this script with --teacher lmstudio
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import threading
import json
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def resolve_output_dir(mode: str, output_dir_arg: str | None) -> Path:
    """출력 디렉토리를 결정한다.

    - output_dir_arg가 있으면 해당 경로를 그대로 사용한다.
    - 없으면 data/generated/{mode}/run_{YYYYMMDD}/ 형식으로 생성한다.
      동일 날짜에 이미 존재하면 run_{YYYYMMDD}_2, run_{YYYYMMDD}_3, ... 으로 증가한다.
    디렉토리는 자동으로 생성된다.
    """
    if output_dir_arg is not None:
        out = Path(output_dir_arg)
        out.mkdir(parents=True, exist_ok=True)
        return out

    date_str = time.strftime("%Y%m%d")
    base = Path("data") / "generated" / mode
    candidate = base / f"run_{date_str}"

    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    # 충돌 시 _2, _3, ... 으로 증가
    suffix = 2
    while True:
        candidate = base / f"run_{date_str}_{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        suffix += 1


def merge_to_processed(run_dir: Path, processed_dir: Path) -> int:
    """run_dir의 모든 .jsonl 파일에서 text 필드를 추출하여
    processed_dir/train.txt에 append한다.

    - text 필드 없는 항목은 건너뛰고 경고를 출력한다.
    - processed_dir이 없으면 자동으로 생성한다.
    - 반환값: 병합된 항목 수
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_txt = processed_dir / "train.txt"

    merged = 0
    with open(train_txt, "a", encoding="utf-8") as out_f:
        for jsonl_file in sorted(run_dir.glob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as in_f:
                for lineno, raw in enumerate(in_f, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError as e:
                        print(f"  ⚠ [{jsonl_file.name}:{lineno}] JSON 파싱 실패: {e}")
                        continue
                    if "text" not in item:
                        print(f"  ⚠ [{jsonl_file.name}:{lineno}] 'text' 필드 없음 — 건너뜀")
                        continue
                    out_f.write(item["text"] + "\n")
                    merged += 1

    return merged


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPTS = [
    # Conversations
    "두 친구가 주말 계획에 대해 이야기하는 자연스러운 한국어 대화를 써줘. 약 10줄.",
    "엄마와 딸이 저녁 식사 준비하면서 나누는 따뜻한 대화를 써줘. 약 10줄.",
    "카페에서 손님과 바리스타가 나누는 일상적인 대화를 써줘. 약 8줄.",
    "친구들이 영화를 보고 나서 감상을 이야기하는 대화를 써줘. 약 10줄.",
    "선생님과 학생이 수업 후에 나누는 대화를 써줘. 약 8줄.",
    "두 직장 동료가 점심시간에 나누는 대화를 써줘. 약 10줄.",
    "부모님과 자녀가 여행 계획을 세우는 대화를 써줘. 약 10줄.",
    "친구가 고민을 털어놓고 위로받는 대화를 써줘. 따뜻하고 공감적인 내용으로. 약 12줄.",
    "형제자매가 장난치며 나누는 재미있는 대화를 써줘. 약 10줄.",
    "할아버지와 손녀가 옛날 이야기를 나누는 대화를 써줘. 약 10줄.",
    # Stories & narratives
    "봄날 공원을 산책하는 소녀의 이야기를 한국어로 써줘. 약 150단어.",
    "작은 마을에 사는 할머니와 손자의 하루 이야기를 써줘. 약 150단어.",
    "처음으로 혼자 요리에 도전하는 청년의 이야기를 써줘. 약 150단어.",
    "고양이가 집에서 하루를 보내는 이야기를 귀엽게 써줘. 약 100단어.",
    "비 오는 날 도서관에서 책을 읽는 사람의 이야기를 써줘. 약 120단어.",
    "첫 출근날 긴장한 신입사원의 이야기를 써줘. 약 150단어.",
    "길을 잃은 강아지가 집을 찾아가는 이야기를 써줘. 약 130단어.",
    "오래된 편지를 발견한 할머니의 이야기를 써줘. 약 150단어.",
    # Korean culture & knowledge
    "한국의 사계절 특징을 자연스러운 한국어로 설명해줘. 약 150단어.",
    "김치 담그는 방법을 쉽고 친근하게 설명해줘. 약 150단어.",
    "한국의 명절 추석에 대해 설명해줘. 약 150단어.",
    "한복의 아름다움과 특징에 대해 설명해줘. 약 120단어.",
    "K-pop이 왜 세계적으로 인기를 끌게 됐는지 설명해줘. 약 150단어.",
    "한국 전통 음식 비빔밥에 대해 소개해줘. 약 120단어.",
    "한국의 교육 문화에 대해 설명해줘. 약 150단어.",
    "서울의 유명한 관광지를 소개하는 글을 써줘. 약 150단어.",
    "한국의 찜질방 문화를 소개해줘. 약 120단어.",
    "한국 전통 놀이(윷놀이, 제기차기 등)를 소개해줘. 약 130단어.",
    # Daily life & emotions
    "월요일 아침 출근길의 느낌을 생생하게 묘사해줘. 약 100단어.",
    "좋아하는 음식을 먹을 때의 행복감을 표현하는 글을 써줘. 약 80단어.",
    "오랜 친구를 다시 만났을 때의 기쁨을 표현해줘. 약 100단어.",
    "시험을 앞두고 긴장한 학생의 심정을 써줘. 약 100단어.",
    "처음 강아지를 키우게 된 날의 설렘을 써줘. 약 100단어.",
    "퇴근 후 집에 돌아왔을 때의 편안함을 표현해줘. 약 80단어.",
    "첫사랑을 떠올리는 감정을 글로 표현해줘. 약 100단어.",
    "졸업식 날의 설레고 아쉬운 마음을 글로 써줘. 약 100단어.",
    # Q&A / Educational
    "한국어에서 존댓말과 반말의 차이를 예시와 함께 설명해줘. 약 150단어.",
    "한국의 젓가락 사용 예절에 대해 설명해줘. 약 120단어.",
    "한국어 숫자 체계(한자어/고유어)를 간단히 설명해줘. 약 120단어.",
    "'정(情)'이라는 한국 문화 개념을 설명해줘. 약 130단어.",
    "한국에서 나이를 물어보는 문화에 대해 설명해줘. 약 100단어.",
    "한국어 인사말의 다양한 표현을 소개해줘. 약 120단어.",
    # Poems & creative
    "봄, 여름, 가을, 겨울을 주제로 짧은 한국어 시 4편을 써줘.",
    "가족을 주제로 따뜻한 짧은 시를 써줘.",
    "고향을 그리워하는 마음을 담은 짧은 글을 써줘. 약 80단어.",
    "새벽 하늘을 바라보며 느끼는 감정을 글로 표현해줘. 약 80단어.",
    "바다를 주제로 자유시를 써줘. 약 80단어.",
    # Descriptions
    "한국의 전통 시장(재래시장) 풍경을 생생하게 묘사해줘. 약 130단어.",
    "한강 공원의 주말 풍경을 묘사해줘. 약 130단어.",
    "학교 운동장에서 아이들이 노는 풍경을 묘사해줘. 약 100단어.",
    "깊은 가을 숲 속 풍경을 아름답게 묘사해줘. 약 100단어.",
    "눈 오는 날 거리 풍경을 묘사해줘. 약 100단어.",
]

SYSTEM = (
    "당신은 고품질 한국어 텍스트를 생성하는 교사입니다. "
    "자연스럽고 유창한 현대 한국어로 응답하세요. "
    "마크다운 형식 없이 순수 텍스트만 출력하세요."
)


QA_SYSTEM = (
    "당신은 유능하고 친절한 한국어 AI 어시스턴트입니다. "
    "사용자의 질문에 대해 정확하고 자연스러운 한국어로 상세하게 답변하세요. "
    "마크다운 형식 없이 순수 텍스트만 출력하세요."
)


# ---------------------------------------------------------------------------
# Backend clients
# ---------------------------------------------------------------------------

def make_gemini_client(api_key: str):
    import google.genai as genai
    return genai.Client(api_key=api_key)


def make_deepseek_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def make_mistral_client(api_key: str):
    from mistralai.client import Mistral
    return Mistral(api_key=api_key)


def make_openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def generate_gemini(client, prompt: str, model: str = "gemini-2.5-flash", system_prompt: str = SYSTEM, retries: int = 6) -> str | None:
    from google.genai import types
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.9,
                    max_output_tokens=600,
                ),
            )
            return resp.text.strip()
        except Exception as e:
            is_503 = "503" in str(e) or "UNAVAILABLE" in str(e)
            wait = (5 * (attempt + 1)) if is_503 else (2 ** attempt)
            print(f"  [Gemini] ⚠ Attempt {attempt+1} failed: {str(e)[:80]}. Retry in {wait}s...")
            time.sleep(wait)
    return None


def generate_deepseek(client, prompt: str, system_prompt: str = SYSTEM, retries: int = 6) -> str | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=600,
                stream=False,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [DeepSeek] ⚠ Attempt {attempt+1} failed: {str(e)[:80]}. Retry in {wait}s...")
            time.sleep(wait)
    return None


def generate_mistral(client, prompt: str, system_prompt: str = SYSTEM, retries: int = 6) -> str | None:
    for attempt in range(retries):
        try:
            resp = client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=600,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [Mistral] ⚠ Attempt {attempt+1} failed: {str(e)[:80]}. Retry in {wait}s...")
            time.sleep(wait)
    return None


def generate_openai(client, prompt: str, system_prompt: str = SYSTEM, retries: int = 6) -> str | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=600,
                stream=False,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [OpenAI] ⚠ Attempt {attempt+1} failed: {str(e)[:80]}. Retry in {wait}s...")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# DEGS: Dynamic Evolutive Generation System
# ---------------------------------------------------------------------------

DIRECTOR_SYSTEM = (
    "You are the Director of a Korean language dataset project. "
    "Your goal is to generate diverse, interesting, and culturally rich prompts for other AI models to answer in Korean. "
    "Focus on variety: technical, casual, historical, slang, emotional, and specialized domains. "
    "Output only the prompts, one per line."
)

QA_DIRECTOR_SYSTEM = (
    "You are the Director of a Korean language Question/Answer dataset project. "
    "Your goal is to generate diverse and challenging questions IN KOREAN that a helpful AI assistant should be able to answer. "
    "IMPORTANT: Questions must cover a WIDE range of topics. Do NOT focus on Korean culture. "
    "Distribute questions evenly across these categories: "
    "science (physics, chemistry, biology, astronomy), "
    "mathematics and logic puzzles, "
    "world history and geography, "
    "technology and programming, "
    "philosophy and ethics, "
    "everyday practical advice, "
    "creative writing prompts, "
    "economics and society, "
    "health and medicine, "
    "Korean language and culture (max 1 out of 10 questions). "
    "Output only the questions in Korean, one per line, no numbering."
)

REFEREE_SYSTEM = (
    "You are a Referee for Korean text quality. "
    "Rate the following Korean text on a scale of 0.0 to 1.0 based on naturalness, grammar, and depth. "
    "Output ONLY a JSON object: {\"score\": 0.85, \"reason\": \"explanation\"}"
)

def generate_dynamic_prompts_gemini(client, model: str = "gemini-2.5-flash", count: int = 20, mode: str = "text") -> list[str]:
    """Use Gemini as Director to generate new prompt ideas."""
    print(f"🎬 [Director/Gemini] Generating {count} new {mode} ideas...")
    from google.genai import types

    system_instruction = DIRECTOR_SYSTEM if mode == "text" else QA_DIRECTOR_SYSTEM
    topic = "prompts to elicit high-quality Korean text" if mode == "text" else "challenging and diverse questions for a Korean AI"

    try:
        prompt = (
            f"Generate {count} unique and creative {topic}. "
            "Mix categories: everyday life, philosophy, science fiction, historical drama, modern slang, and professional emails."
        )
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=1.0,
            ),
        )
        new_prompts = [line.strip() for line in resp.text.strip().splitlines() if line.strip()]
        return new_prompts[:count]
    except Exception as e:
        print(f"  [Director/Gemini] ⚠ Failed: {e}")
        return []


def generate_dynamic_prompts_deepseek(client, count: int = 20, mode: str = "text") -> list[str]:
    """Use DeepSeek as Director to generate new prompt ideas."""
    print(f"🎬 [Director/DeepSeek] Generating {count} new {mode} ideas...")

    system_instruction = DIRECTOR_SYSTEM if mode == "text" else QA_DIRECTOR_SYSTEM
    topic = "prompts to elicit high-quality Korean text" if mode == "text" else "challenging and diverse questions for a Korean AI"

    try:
        prompt = (
            f"Generate {count} unique and creative {topic}. "
            "Mix categories: everyday life, philosophy, science fiction, history, coding, math, logic, and general knowledge. "
            "Output only the prompts/questions, one per line, no numbering."
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            max_tokens=800,
            stream=False,
        )
        text = resp.choices[0].message.content.strip()
        new_prompts = [line.strip() for line in text.splitlines() if line.strip()]
        return new_prompts[:count]
    except Exception as e:
        print(f"  [Director/DeepSeek] ⚠ Failed: {e}")
        return []


def generate_dynamic_prompts_mistral(client, count: int = 20, mode: str = "text") -> list[str]:
    """Use Mistral as Director to generate new prompt ideas."""
    print(f"🎬 [Director/Mistral] Generating {count} new {mode} ideas...")

    system_instruction = DIRECTOR_SYSTEM if mode == "text" else QA_DIRECTOR_SYSTEM
    topic = "prompts to elicit high-quality Korean text" if mode == "text" else "challenging and diverse questions for a Korean AI"

    try:
        prompt = (
            f"Generate {count} unique and creative {topic}. "
            "Mix categories: everyday life, philosophy, science fiction, history, coding, math, logic, and general knowledge. "
            "Output only the prompts/questions, one per line, no numbering."
        )
        resp = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            max_tokens=800,
        )
        text = resp.choices[0].message.content.strip()
        new_prompts = [line.strip() for line in text.splitlines() if line.strip()]
        return new_prompts[:count]
    except Exception as e:
        print(f"  [Director/Mistral] ⚠ Failed: {e}")
        return []


def generate_dynamic_prompts_openai(client, count: int = 20, mode: str = "text") -> list[str]:
    """Use OpenAI as Director to generate new prompt ideas."""
    print(f"🎬 [Director/OpenAI] Generating {count} new {mode} ideas...")

    system_instruction = DIRECTOR_SYSTEM if mode == "text" else QA_DIRECTOR_SYSTEM
    topic = "prompts to elicit high-quality Korean text" if mode == "text" else "challenging and diverse questions for a Korean AI"

    try:
        prompt = (
            f"Generate {count} unique and creative {topic}. "
            "Mix categories: everyday life, philosophy, science fiction, history, coding, math, logic, and general knowledge. "
            "Output only the prompts/questions, one per line, no numbering."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            max_tokens=800,
            stream=False,
        )
        text = resp.choices[0].message.content.strip()
        new_prompts = [line.strip() for line in text.splitlines() if line.strip()]
        return new_prompts[:count]
    except Exception as e:
        print(f"  [Director/OpenAI] ⚠ Failed: {e}")
        return []


def generate_dynamic_prompts(client, model: str = "gemini-2.5-flash", count: int = 20, mode: str = "text") -> list[str]:
    """Gemini Director wrapper (하위 호환용)."""
    return generate_dynamic_prompts_gemini(client, model=model, count=count, mode=mode)

def rate_quality(client, text: str, model: str = "gemini-2.5-flash") -> dict:
    """Use the Referee model to score the generated text."""
    from google.genai import types
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [{"text": f"Rate this Korean text:\n\n{text}"}]}],
            config=types.GenerateContentConfig(
                system_instruction=REFEREE_SYSTEM,
                response_mime_type="application/json",
            ),
        )
        return json.loads(resp.text)
    except Exception:
        return {"score": 0.5, "reason": "Scoring failed"}


def run_worker(
    name: str,
    generate_fn,
    client,
    prompts: list[str],
    batches: int,
    jsonl_path: Path,
    delay: float,
    lock: threading.Lock,
    stats: dict,
    referee_client=None,
    mode: str = "text",
    gemini_model: str = "gemini-2.5-flash",
    director_fn=None,
    director_client=None,
) -> None:
    """Worker thread to generate and save data with evaluation.
    
    batches=0 means infinite loop (Ctrl-C to stop).
    director_fn: callable to refresh prompts periodically (for infinite mode).
    """
    infinite = (batches == 0)
    mode_str = "∞" if infinite else str(batches)
    print(f"🚀 [{name}] Started worker (Mode: {mode}, Batches: {mode_str})...")
    
    system_prompt = SYSTEM if mode == "text" else QA_SYSTEM
    prompt_pool = list(prompts)
    used_prompts = set()
    refresh_interval = 30  # 30개마다 새 프롬프트 생성
    
    i = 0
    while True:
        if not infinite and i >= batches:
            break
        i += 1

        # 주기적으로 Director에게 새 프롬프트 요청 (유한/무한 모드 모두)
        need_refresh = (i % refresh_interval == 1) and director_fn and director_client
        if need_refresh:
            print(f"  [{name}] 🔄 Refreshing prompts (batch {i})...")
            try:
                new_prompts = director_fn(director_client, count=20, mode=mode)
                if new_prompts:
                    # 이미 사용한 프롬프트 제외하고 추가
                    fresh = [p for p in new_prompts if p not in used_prompts]
                    if fresh:
                        prompt_pool = fresh
                        print(f"  [{name}] ✨ Got {len(fresh)} fresh prompts")
                    else:
                        # 전부 중복이면 그냥 새 풀로 교체 (반복보다 낫다)
                        prompt_pool = new_prompts
                        print(f"  [{name}] ♻ All new prompts already used, replacing pool anyway")
            except Exception as e:
                print(f"  [{name}] ⚠ Director refresh failed: {e}")

        # 풀에서 아직 안 쓴 프롬프트 우선 선택
        unused = [p for p in prompt_pool if p not in used_prompts]
        if unused:
            prompt = random.choice(unused)
        else:
            # 전부 사용했으면 director에게 즉시 새 프롬프트 요청
            if director_fn and director_client:
                try:
                    new_prompts = director_fn(director_client, count=20, mode=mode)
                    fresh = [p for p in new_prompts if p not in used_prompts]
                    if fresh:
                        prompt_pool = fresh
                        prompt = random.choice(fresh)
                        print(f"  [{name}] ✨ Pool exhausted — got {len(fresh)} new prompts")
                    else:
                        # 정말 다 썼으면 used_prompts 절반 초기화 후 재사용
                        used_prompts = set(list(used_prompts)[len(used_prompts)//2:])
                        prompt = random.choice(prompt_pool)
                        print(f"  [{name}] 🔁 Resetting half of used_prompts to allow reuse")
                except Exception:
                    prompt = random.choice(prompt_pool)
            else:
                prompt = random.choice(prompt_pool)
        used_prompts.add(prompt)

        batch_label = f"{i:4d}/∞" if infinite else f"{i:4d}/{batches}"
        print(f"  [{name}] [{batch_label}] {prompt[:55]}...")
        
        # Add gemini_model to generate_fn if it's Gemini
        if "gemini" in name.lower():
            text = generate_fn(client, prompt, model=gemini_model, system_prompt=system_prompt)
        else:
            text = generate_fn(client, prompt, system_prompt=system_prompt)
        
        if text:
            score_data = {"score": 1.0}
            if referee_client and i % 5 == 0:  # Sample scoring to save quota
                score_data = rate_quality(referee_client, text, model=gemini_model)
            
            entry = {
                "text": text,
                "prompt": prompt,
                "mode": mode,
                "teacher": name.lower(),
                "score": score_data.get("score", 1.0),
                "reason": score_data.get("reason", ""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }

            # qa 모드: question, answer 필드 추가
            if mode == "qa":
                entry["question"] = prompt
                entry["answer"] = text
            
            with lock:
                # Save as JSONL only
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
                stats["success"] += 1
                stats["total_chars"] += len(text)
                print(f"  [{name}] ✅ {len(text)} chars (Score: {score_data.get('score', 'N/A')})")
        else:
            with lock:
                stats["failed"] += 1
                print(f"  [{name}] ❌ Failed (skipping)")
        
        if delay > 0:
            time.sleep(delay)

    print(f"\n✅ [{name}] Done! ({i} batches completed)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Generate Korean training data using AI teachers")
    parser.add_argument("--output-dir", default=None, help="Output directory for data files (default: data/generated/{mode}/run_{YYYYMMDD}/)")
    parser.add_argument("--batches", type=int, default=200, help="Batches per teacher")
    parser.add_argument("--teacher", choices=["gemini", "deepseek", "mistral", "openai", "lmstudio", "both", "all"], default="both")
    parser.add_argument("--mode", choices=["text", "qa"], default="text", help="Generation mode: 'text' (general) or 'qa' (Question/Answer pairs)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between calls per teacher")
    parser.add_argument("--merge", action="store_true", help="생성 완료 후 data/processed/train.txt에 자동 병합")
    args = parser.parse_args()

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    # Ensure it's not the old deprecated name
    if "gemini-2.0-flash" in gemini_model and "preview" not in gemini_model:
        gemini_model = "gemini-2.5-flash"
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    use_gemini = args.teacher in ("gemini", "both", "all")
    use_deepseek = args.teacher in ("deepseek", "both", "all")
    use_mistral = args.teacher in ("mistral", "all")
    use_openai = args.teacher in ("openai", "all")
    use_lmstudio = args.teacher in ("lmstudio",)

    if use_gemini and not gemini_key:
        print("❌ GEMINI_API_KEY missing in .env")
        sys.exit(1)
    if use_deepseek and not deepseek_key:
        print("❌ DEEPSEEK_API_KEY missing in .env")
        print("   Get one free at: https://platform.deepseek.com")
        sys.exit(1)
    if use_mistral and not mistral_key:
        print("❌ MISTRAL_API_KEY missing in .env")
        print("   Get one at: https://console.mistral.ai")
        sys.exit(1)
    if use_openai and not openai_key:
        print("❌ OPENAI_API_KEY missing in .env")
        print("   Get one at: https://platform.openai.com")
        sys.exit(1)

    output_dir = resolve_output_dir(args.mode, args.output_dir)

    # 각 teacher별 JSONL 파일명: {teacher}.jsonl (run 폴더 안에 저장)
    gemini_output = output_dir / "gemini.jsonl"
    deepseek_output = output_dir / "deepseek.jsonl"
    mistral_output = output_dir / "mistral.jsonl"
    openai_output = output_dir / "openai.jsonl"
    lmstudio_output = output_dir / "lmstudio.jsonl"

    teachers_str = args.teacher.upper()

    print(f"🎓 AI Teachers → GrinVi Student")
    print(f"   Teachers : {teachers_str}")
    print(f"   Batches  : {args.batches} per teacher")
    print(f"   Output   : {output_dir}/")
    print()

    lock = threading.Lock()
    stats = {"total_chars": 0, "success": 0, "failed": 0}
    threads = []

    # gemini_client를 None으로 먼저 초기화 (다른 teacher에서 참조 시 안전하게)
    gemini_client = None

    # Dynamic prompt generation (Director) — Gemini 우선, 없으면 DeepSeek fallback
    dynamic_prompts = []
    if use_gemini:
        try:
            gemini_client = make_gemini_client(gemini_key)
            dynamic_prompts = generate_dynamic_prompts_gemini(gemini_client, model=gemini_model, count=30, mode=args.mode)
        except Exception as e:
            print(f"❌ Could not run Gemini Director: {e}")
            gemini_client = None

    if not dynamic_prompts and use_deepseek:
        try:
            _director_ds = make_deepseek_client(deepseek_key)
            dynamic_prompts = generate_dynamic_prompts_deepseek(_director_ds, count=30, mode=args.mode)
        except Exception as e:
            print(f"❌ Could not run DeepSeek Director: {e}")

    if not dynamic_prompts and use_mistral:
        try:
            _director_mi = make_mistral_client(mistral_key)
            dynamic_prompts = generate_dynamic_prompts_mistral(_director_mi, count=30, mode=args.mode)
        except Exception as e:
            print(f"❌ Could not run Mistral Director: {e}")

    if not dynamic_prompts and use_openai:
        try:
            _director_oai = make_openai_client(openai_key)
            dynamic_prompts = generate_dynamic_prompts_openai(_director_oai, count=30, mode=args.mode)
        except Exception as e:
            print(f"❌ Could not run OpenAI Director: {e}")

    prompts = (PROMPTS if args.mode == "text" else []) + dynamic_prompts
    if not prompts:
        # Fallback if no dynamic prompts and not in text mode
        prompts = ["한국의 역사에 대해 설명해줘.", "맛있는 김치찌개 레시피를 알려줘.", "인공지능의 미래는 어떻게 될까?"]
    
    random.shuffle(prompts)

    if use_gemini:
        if gemini_client is None:
            print("❌ Could not init Gemini: client initialization failed")
        else:
            try:
                t = threading.Thread(
                    target=run_worker,
                    args=("Gemini", generate_gemini, gemini_client,
                          prompts, args.batches, gemini_output, args.delay, lock, stats, gemini_client, args.mode, gemini_model),
                    kwargs={"director_fn": generate_dynamic_prompts_gemini, "director_client": gemini_client},
                    daemon=True,
                )
                threads.append(("Gemini", gemini_output, t))
            except Exception as e:
                print(f"❌ Could not init Gemini: {e}")

    if use_deepseek:
        try:
            deepseek_client = make_deepseek_client(deepseek_key)
            shifted_prompts = prompts[len(prompts)//2:] + prompts[:len(prompts)//2]
            t = threading.Thread(
                target=run_worker,
                args=("DeepSeek", generate_deepseek, deepseek_client,
                      shifted_prompts, args.batches, deepseek_output, args.delay, lock, stats,
                      gemini_client, args.mode, gemini_model),
                kwargs={"director_fn": generate_dynamic_prompts_deepseek, "director_client": deepseek_client},
                daemon=True,
            )
            threads.append(("DeepSeek", deepseek_output, t))
        except Exception as e:
            print(f"❌ Could not init DeepSeek: {e}")

    if use_mistral:
        try:
            mistral_client = make_mistral_client(mistral_key)
            shifted_prompts = prompts[len(prompts)//3:] + prompts[:len(prompts)//3]
            t = threading.Thread(
                target=run_worker,
                args=("Mistral", generate_mistral, mistral_client,
                      shifted_prompts, args.batches, mistral_output, args.delay, lock, stats,
                      gemini_client, args.mode, gemini_model),
                kwargs={"director_fn": generate_dynamic_prompts_mistral, "director_client": mistral_client},
                daemon=True,
            )
            threads.append(("Mistral", mistral_output, t))
        except Exception as e:
            print(f"❌ Could not init Mistral: {e}")

    if use_openai:
        try:
            openai_client = make_openai_client(openai_key)
            shifted_prompts = prompts[len(prompts)//5:] + prompts[:len(prompts)//5]
            t = threading.Thread(
                target=run_worker,
                args=("OpenAI", generate_openai, openai_client,
                      shifted_prompts, args.batches, openai_output, args.delay, lock, stats,
                      gemini_client, args.mode, gemini_model),
                kwargs={"director_fn": generate_dynamic_prompts_openai, "director_client": openai_client},
                daemon=True,
            )
            threads.append(("OpenAI", openai_output, t))
        except Exception as e:
            print(f"❌ Could not init OpenAI: {e}")

    if use_lmstudio:
        try:
            from openai import OpenAI as _OpenAI
            lmstudio_base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
            lmstudio_client = _OpenAI(base_url=lmstudio_base_url, api_key="lm-studio")
            # 사용 가능한 모델 중 한국어 가능한 것 선택
            models = lmstudio_client.models.list()
            lmstudio_model = None
            # 환경변수로 모델 강제 지정 가능
            env_model = os.getenv("LMSTUDIO_MODEL", "").strip()
            if env_model:
                lmstudio_model = env_model
            else:
                preferred = ["gemma", "qwen", "mistral", "llama", "deepseek"]
                for pref in preferred:
                    for m in models.data:
                        if pref in m.id.lower():
                            lmstudio_model = m.id
                            break
                    if lmstudio_model:
                        break
            if not lmstudio_model and models.data:
                lmstudio_model = models.data[0].id
            if not lmstudio_model:
                raise ValueError("No models loaded in LM Studio")

            print(f"🖥️  LM Studio model: {lmstudio_model}")

            def generate_lmstudio(client, prompt: str, mode: str = "text", system_prompt: str = None) -> str:
                if system_prompt:
                    sys_msg = system_prompt
                elif mode == "text":
                    sys_msg = "당신은 한국어로 긴 글을 작성하는 전문 작가입니다. 주어진 주제에 대해 상세하고 풍부한 내용으로 장문의 글을 작성하세요."
                else:
                    sys_msg = "당신은 한국어 교육 전문가입니다. 주어진 주제에 대해 질문과 상세한 답변을 작성하세요."
                resp = client.chat.completions.create(
                    model=lmstudio_model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2048,
                    temperature=0.8,
                )
                return resp.choices[0].message.content.strip()

            _lmstudio_used_topics: set[str] = set()

            def generate_dynamic_prompts_lmstudio(client, count=30, mode="text"):
                avoid = ""
                if _lmstudio_used_topics:
                    sample = list(_lmstudio_used_topics)[-20:]  # 최근 20개만 전달
                    avoid = f"\n다음 주제는 이미 사용했으니 제외해줘: {', '.join(sample)}"

                if mode == "text":
                    instruction = (
                        f"한국어 장문 글쓰기 학습 데이터를 위한 다양한 주제 {count}개를 한 줄씩 나열해줘. "
                        "일상, 철학, 사회, 과학, 역사, 문화, 감정, 직업, 환경 등 카테고리를 골고루 섞어줘. "
                        "번호 없이 주제만 출력해줘." + avoid
                    )
                else:
                    instruction = (
                        f"한국어 QA 학습 데이터를 위한 다양한 질문 {count}개를 한 줄씩 나열해줘. "
                        "과학, 역사, 기술, 철학, 일상, 언어, 수학, 사회 등 카테고리를 골고루 섞어줘. "
                        "번호 없이 질문만 출력해줘." + avoid
                    )

                try:
                    resp = client.chat.completions.create(
                        model=lmstudio_model,
                        messages=[{"role": "user", "content": instruction}],
                        max_tokens=1024,
                        temperature=1.0,
                    )
                    lines = resp.choices[0].message.content.strip().split("\n")
                    result = [l.strip().lstrip("0123456789.-) ") for l in lines if l.strip()][:count]
                    _lmstudio_used_topics.update(result)
                    return result
                except Exception as e:
                    print(f"  [LMStudio Director] ⚠ {e}")
                    return []

            shifted_prompts = prompts[len(prompts)//7:] + prompts[:len(prompts)//7]
            t = threading.Thread(
                target=run_worker,
                args=("LMStudio", generate_lmstudio, lmstudio_client,
                      shifted_prompts, args.batches, lmstudio_output, args.delay, lock, stats,
                      gemini_client, args.mode, gemini_model),
                kwargs={"director_fn": generate_dynamic_prompts_lmstudio, "director_client": lmstudio_client},
                daemon=True,
            )
            threads.append(("LMStudio", lmstudio_output, t))
        except Exception as e:
            print(f"❌ Could not init LM Studio: {e}")

    if not threads:
        print("❌ No teachers enabled!")
        sys.exit(1)

    # Start all threads
    for name, output_file, t in threads:
        t.start()
    
    # Wait for all threads to complete
    for name, output_file, t in threads:
        t.join()

    print(f"\n{'='*50}")
    print(f"✅ All teachers done!")
    print(f"   ✔ Success : {stats['success']} batches")
    print(f"   ✘ Failed  : {stats['failed']} batches")
    print(f"   📝 Total  : {stats['total_chars']:,} characters")
    print()
    print(f"📂 Generated files in: {output_dir}/")
    for name, output_file, t in threads:
        if output_file.exists():
            size = output_file.stat().st_size
            print(f"   ✓ {output_file.name} ({size:,} bytes)")

    # --merge 옵션 처리
    if args.merge:
        processed_dir = Path("data") / "processed"
        print(f"\n🔀 Merging to {processed_dir / 'train.txt'} ...")
        merged_count = merge_to_processed(output_dir, processed_dir)
        print(f"   ✅ {merged_count}개 항목 병합 완료 → {processed_dir / 'train.txt'}")
    
    print(f"\n🎓 Train GrinVi:")
    print(f"   python scripts/train.py --preset tiny \\")
    print(f"       --tokenizer morph \\")
    print(f"       --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \\")
    print(f"       --data data/processed/train.txt \\")
    print(f"       --max_steps 100000000 --grad_ckpt")


if __name__ == "__main__":
    main()

