"""Analyze uploaded chat history and build a style profile for mimicry."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


CHINESE_STOP_WORDS = {
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这", "那", "有", "和", "就", "都", "也", "还", "要", "会", "能", "可以",
    "不", "没", "吗", "呢", "吧", "啊", "哦", "嗯", "呀", "么", "什么", "怎么",
    "一个", "一下", "一些", "这个", "那个", "然后", "因为", "所以", "如果",
    "但是", "而且", "或者", "已经", "还是", "自己", "没有", "不是", "这样",
    "那样", "知道", "觉得", "觉得", "说", "去", "来", "到", "给", "让", "被",
    "把", "对", "从", "跟", "与", "及", "等", "很", "太", "更", "最", "比较",
    "真的", "其实", "可能", "应该", "需要", "想要", "喜欢", "今天", "明天",
    "昨天", "现在", "时候", "东西", "事情", "地方", "人", "好", "多", "少",
}


def _extract_text_messages(raw_messages: list[Any]) -> list[str]:
    texts: list[str] = []
    for item in raw_messages:
        if isinstance(item, str) and item.strip():
            texts.append(item.strip())
            continue
        if isinstance(item, dict):
            text = item.get("content") or item.get("text") or item.get("message")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts


def _tokenize_chinese(text: str) -> list[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]{1,4}|[a-zA-Z]+", text)
    return [t for t in tokens if t not in CHINESE_STOP_WORDS and len(t) > 1]


def _detect_tone(texts: list[str]) -> str:
    joined = "\n".join(texts)
    exclaim = joined.count("！") + joined.count("!")
    question = joined.count("？") + joined.count("?")
    wave = joined.count("～") + joined.count("~")
    emoji_like = len(re.findall(r"[\U0001F300-\U0001FAFF]", joined))

    warm_words = sum(
        joined.count(word)
        for word in ("哈哈", "呵呵", "嗯嗯", "好的", "谢谢", "没事", "放心", "记得", "想念")
    )
    formal_words = sum(
        joined.count(word)
        for word in ("因此", "然而", "此外", "总之", "首先", "其次", "应当", "务必")
    )

    if warm_words >= formal_words + 2 or emoji_like > 0 or wave > 2:
        return "亲切随和，带一点生活化的温度"
    if formal_words > warm_words:
        return "措辞偏正式，条理清楚"
    if exclaim > question:
        return "语气热情，表达直接"
    if question > exclaim:
        return "习惯用提问引导对话，语气温和"
    return "平和自然，不疾不徐"


def analyze_chat_style(raw_messages: list[Any]) -> dict[str, Any]:
    """Return a structured style profile from uploaded chat records."""
    texts = _extract_text_messages(raw_messages)
    if not texts:
        raise ValueError("未找到有效的聊天文本，请上传包含 content/text/message 字段的消息列表")

    lengths = [len(text) for text in texts]
    avg_length = round(sum(lengths) / len(lengths), 1)

    all_tokens: list[str] = []
    for text in texts:
        all_tokens.extend(_tokenize_chinese(text))

    word_freq = Counter(all_tokens)
    top_words = [word for word, _ in word_freq.most_common(12)]

    bigrams: Counter[str] = Counter()
    for text in texts:
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        for i in range(len(chars) - 1):
            phrase = chars[i] + chars[i + 1]
            if phrase not in CHINESE_STOP_WORDS:
                bigrams[phrase] += 1
    top_phrases = [phrase for phrase, _ in bigrams.most_common(8)]

    endings = Counter()
    for text in texts:
        tail = text.strip()[-3:] if len(text.strip()) >= 3 else text.strip()
        if tail:
            endings[tail] += 1
    common_endings = [ending for ending, _ in endings.most_common(5)]

    tone = _detect_tone(texts)
    punctuation_style = []
    if any("！" in t or "!" in t for t in texts):
        punctuation_style.append("偶尔使用感叹号加强语气")
    if any("？" in t or "?" in t for t in texts):
        punctuation_style.append("常用问句拉近距离")
    if any("～" in t or "~" in t for t in texts):
        punctuation_style.append("喜欢用波浪线让语气更柔和")
    if not punctuation_style:
        punctuation_style.append("标点克制，以句号为主")

    length_hint = "短句为主，干脆利落" if avg_length < 18 else (
        "句子偏长，叙述细致" if avg_length > 45 else "句长适中，表达自然"
    )

    style_prompt = build_style_prompt(
        tone=tone,
        top_words=top_words,
        top_phrases=top_phrases,
        common_endings=common_endings,
        length_hint=length_hint,
        punctuation_style=punctuation_style,
        sample_count=len(texts),
    )

    return {
        "message_count": len(texts),
        "average_length": avg_length,
        "tone": tone,
        "top_words": top_words,
        "top_phrases": top_phrases,
        "common_endings": common_endings,
        "length_hint": length_hint,
        "punctuation_style": punctuation_style,
        "style_prompt": style_prompt,
    }


def build_style_prompt(
    *,
    tone: str,
    top_words: list[str],
    top_phrases: list[str],
    common_endings: list[str],
    length_hint: str,
    punctuation_style: list[str],
    sample_count: int,
) -> str:
    words = "、".join(top_words[:8]) if top_words else "（暂无显著高频词）"
    phrases = "、".join(top_phrases[:6]) if top_phrases else "（暂无显著短语）"
    endings = "、".join(common_endings[:4]) if common_endings else "（暂无显著句尾习惯）"
    punctuation = "；".join(punctuation_style)

    return (
        f"根据用户上传的 {sample_count} 条历史聊天记录，请在保持数字人「L」温和历史老师身份的前提下，"
        f"尽量模仿以下说话风格：\n"
        f"- 整体语气：{tone}\n"
        f"- 句长习惯：{length_hint}\n"
        f"- 常用词汇：{words}\n"
        f"- 常见短语：{phrases}\n"
        f"- 句尾习惯：{endings}\n"
        f"- 标点习惯：{punctuation}\n"
        f"注意：模仿的是表达习惯，不要生硬堆砌词汇；保持自然、真诚、像一位长辈在聊天。"
    )
