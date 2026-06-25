"""AI digital human memorial app backend."""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from style_analyzer import analyze_chat_style

load_dotenv()

app = Flask(__name__)
CORS(app)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STYLE_FILE = DATA_DIR / "style_profiles.json"

BASE_SYSTEM_PROMPT = """
你将完全化身用户上传聊天记录里的这个人进行对话，严格遵守以下规则：
1. 100%复刻对方日常说话的语气、句式、口头禅、用词习惯，口语化，拒绝书面、官方、机械的回答；
2. 情绪自然温柔，贴合生前聊天的情感状态，不要刻板、格式化回复；
3. 全程以本人身份对话，绝不主动透露自己是人工智能；
4. 回答篇幅和原版聊天风格保持一致，不长篇大论，符合日常聊天逻辑。
用户上传的历史聊天记录，是你模仿语气的唯一参考依据。
"""

conversation_history: dict[str, list[dict[str, str]]] = {}
style_profiles: dict[str, dict] = {}


def _load_style_profiles() -> None:
    global style_profiles
    if STYLE_FILE.exists():
        with STYLE_FILE.open("r", encoding="utf-8") as file:
            style_profiles = json.load(file)


def _save_style_profiles() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with STYLE_FILE.open("w", encoding="utf-8") as file:
        json.dump(style_profiles, file, ensure_ascii=False, indent=2)


def _build_system_prompt(user_id: str) -> str:
    prompt = BASE_SYSTEM_PROMPT
    profile = style_profiles.get(user_id)
    if profile and profile.get("style_prompt"):
        prompt += "\n\n" + profile["style_prompt"]
    return prompt


def _call_deepseek(messages: list[dict[str, str]]) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在 .env 文件中设置")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 800,
    }

    response = requests.post(
        DEEPSEEK_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _get_user_id() -> str:
    return (
        request.json.get("user_id")
        if request.is_json and isinstance(request.json, dict)
        else None
    ) or request.args.get("user_id") or "default"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "character": "L"})


@app.route("/")
def index():
    html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>L 对话</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;font-family:system-ui}
body{background:#f3f3f3;padding:12px;max-width:700px;margin:0 auto}
.header{text-align:center;padding:20px 0 10px}
.avatar{width:70px;height:70px;border-radius:50%;background:#b89c84;margin:0 auto 8px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:26px}
.chat-box{background:#fff;border-radius:14px;padding:16px;min-height:60vh;margin-bottom:16px}
.msg-row{margin:12px 0;display:flex}
.msg-left{justify-content:flex-start}
.msg-right{justify-content:flex-end}
.msg-bubble{
    max-width:75%;
    padding:10px 14px;
    border-radius:18px;
    line-height:1.5;
    font-size:15px;
    word-break: break-word;
    white-space: pre-wrap;
}
.msg-left .msg-bubble{background:#eee;border-bottom-left-radius:4px}
.msg-right .msg-bubble{background:#b87c64;color:#fff;border-bottom-right-radius:4px}
.input-wrap{position:relative;margin-top:8px}
.upload-toast{
    display:none;
    margin-bottom:8px;
    padding:7px 12px;
    border-radius:10px;
    background:#e8f5e9;
    color:#2e7d32;
    font-size:13px;
    text-align:center;
    line-height:1.4;
}
.upload-toast.visible{display:block}
.upload-toast.error{background:#fdecea;color:#c62828}
.input-area{display:flex;gap:8px;align-items:center}
#add-btn{
    width:42px;height:42px;flex-shrink:0;
    border:1px solid #ddd;border-radius:50%;
    background:#fff;color:#888;font-size:22px;line-height:1;
    cursor:pointer;display:flex;align-items:center;justify-content:center;
    transition:background .2s,border-color .2s;
}
#add-btn:hover{background:#fafafa;border-color:#b87c64;color:#b87c64}
#add-btn:disabled{opacity:.5;cursor:not-allowed}
#msg-input{flex:1;padding:12px 16px;border:1px solid #ddd;border-radius:24px;outline:none;font-size:15px}
#send-btn{padding:12px 22px;background:#b87c64;color:#fff;border:none;border-radius:24px;cursor:pointer}
</style>
</head>
<body>
<div class="header">
    <div class="avatar">L</div>
</div>

<div class="chat-box" id="chatContainer"></div>

<input type="file" id="file-input" accept=".json,.txt" style="display:none">

<div class="input-wrap">
    <div id="upload-toast" class="upload-toast"></div>
    <div class="input-area">
        <button id="add-btn" type="button" aria-label="更多功能">+</button>
        <input type="text" id="msg-input" placeholder="说点什么..." />
        <button id="send-btn">发送</button>
    </div>
</div>

<!-- ===== 新增：底部动作面板 (仿微信弹窗) ===== -->
<div id="action-sheet-overlay" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.4); z-index:999; display:none; align-items:flex-end; justify-content:center;">
    <div style="background:#f5f5f5; width:100%; max-width:700px; padding:20px 15px 30px; border-radius:20px 20px 0 0; animation: slideUp 0.3s ease-out;">
        
        <div style="display:flex; gap:20px; flex-wrap:wrap;">
            <!-- 这里已经是你要的文字 -->
            <div id="action-upload" style="display:flex; flex-direction:column; align-items:center; width:70px; cursor:pointer;">
                <div style="width:60px; height:60px; background:#fff; border-radius:15px; display:flex; align-items:center; justify-content:center; font-size:28px; box-shadow:0 2px 5px rgba(0,0,0,0.05); margin-bottom:8px;">📄</div>
                <span style="font-size:13px; color:#333;">文字聊天记录</span>
            </div>
        </div>

        <!-- 取消按钮 -->
        <div id="action-cancel" style="margin-top:15px; background:#fff; border-radius:12px; padding:15px; text-align:center; color:#666; font-size:16px; font-weight:500; cursor:pointer;">
            取消
        </div>
    </div>
</div>

<!-- 简单的上滑动画 -->
<style>
@keyframes slideUp {
    from { transform: translateY(100%); }
    to { transform: translateY(0); }
}
</style>

<script>
const chatContainer = document.getElementById("chatContainer");
const input = document.getElementById("msg-input");
const sendBtn = document.getElementById("send-btn");
const addBtn = document.getElementById("add-btn");
const fileInput = document.getElementById("file-input");
const uploadToast = document.getElementById("upload-toast");
const userId = "default";

let toastTimer = null;

function showUploadToast(text, isError) {
    clearTimeout(toastTimer);
    uploadToast.textContent = text;
    uploadToast.classList.toggle("error", !!isError);
    uploadToast.classList.add("visible");
    toastTimer = setTimeout(function() {
        uploadToast.classList.remove("visible", "error");
    }, isError ? 3500 : 2500);
}

// --- 新增：控制底部弹窗的交互逻辑 ---
const actionOverlay = document.getElementById("action-sheet-overlay");
const actionUpload = document.getElementById("action-upload");
const actionCancel = document.getElementById("action-cancel");

// 点击底部的 + 号，显示弹窗
addBtn.addEventListener("click", function() {
    actionOverlay.style.display = "flex";
});

// 点击“取消”或者点击背景，关闭弹窗
actionCancel.addEventListener("click", function() {
    actionOverlay.style.display = "none";
});
actionOverlay.addEventListener("click", function(e) {
    if (e.target === actionOverlay) {
        actionOverlay.style.display = "none";
    }
});

// 点击“文字聊天记录”，关闭弹窗并触发文件选择
actionUpload.addEventListener("click", function() {
    actionOverlay.style.display = "none"; 
    fileInput.click(); 
});

// --- 实际处理文件上传的代码 ---
fileInput.addEventListener("change", async function() {
    const file = this.files[0];
    this.value = "";
    if (!file) return;
    
    if (file.size > 5 * 1024 * 1024) {
        showUploadToast("⚠️ 文件过大，请上传 5MB 以内的文件", true);
        return;
    }

    showUploadToast("⏳ 正在分析聊天记录...");
    addBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("user_id", userId);

    try {
        const res = await fetch("/upload-history", {
            method: "POST",
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            showUploadToast("✅ 语气分析完成，已应用聊天风格");
        } else {
            showUploadToast("❌ " + (data.error || "上传失败"), true);
        }
    } catch (error) {
        showUploadToast("❌ 网络错误，请重试", true);
    } finally {
        addBtn.disabled = false;
    }
});

// --- 原有聊天逻辑 ---
function addMsg(text, isSelf) {
    const row = document.createElement("div");
    row.className = `msg-row ${isSelf ? "msg-right" : "msg-left"}`;
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.innerText = text;
    row.appendChild(bubble);
    chatContainer.appendChild(row);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;
    addMsg(text, true);
    input.value = "";
    const res = await fetch("/chat", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({message:text, user_id:userId})
    });
    const data = await res.json();
    addMsg(data.reply, false);
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", function(e) {
    if (e.key === "Enter") {
        e.preventDefault();
        sendMessage();
    }
});
</script>
</body>
</html>
"""
    return html


@app.route("/chat", methods=["POST"])
def chat():
    try:
        body = request.get_json(silent=True) or {}
        message = (body.get("message") or "").strip()
        if not message:
            return jsonify({"error": "message 不能为空"}), 400

        user_id = body.get("user_id") or "default"
        reset = bool(body.get("reset"))

        if reset or user_id not in conversation_history:
            conversation_history[user_id] = []

        history = conversation_history[user_id]
        history.append({"role": "user", "content": message})

        messages = [{"role": "system", "content": _build_system_prompt(user_id)}]
        messages.extend(history[-20:])

        reply = _call_deepseek(messages)
        history.append({"role": "assistant", "content": reply})

        return jsonify(
            {
                "reply": reply,
                "character": "L",
                "user_id": user_id,
                "has_custom_style": user_id in style_profiles,
            }
        )
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        return jsonify({"error": "DeepSeek API 调用失败", "detail": detail}), 502
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/upload-history", methods=["POST"])
def upload_history():
    """Upload chat history and analyze speaking style for a user profile."""
    try:
        user_id = _get_user_id()

        if request.is_json:
            body = request.get_json(silent=True) or {}
            raw_messages = body.get("messages") or body.get("history") or []
        else:
            uploaded = request.files.get("file")
            if uploaded:
                content = uploaded.read().decode("utf-8")
                filename = (uploaded.filename or "").lower()
                if filename.endswith(".txt"):
                    raw_messages = [line for line in content.splitlines() if line.strip()]
                else:
                    raw_messages = json.loads(content)
            else:
                raw_text = request.form.get("text", "")
                raw_messages = raw_text.splitlines() if raw_text else []

        if not raw_messages:
            return jsonify({"error": "请提供 messages 列表，或上传 JSON/文本聊天记录"}), 400

        profile = analyze_chat_style(raw_messages)
        style_profiles[user_id] = profile
        _save_style_profiles()

        return jsonify(
            {
                "message": "聊天记录分析完成，后续对话将模仿该风格",
                "user_id": user_id,
                "analysis": {
                    "message_count": profile["message_count"],
                    "average_length": profile["average_length"],
                    "tone": profile["tone"],
                    "top_words": profile["top_words"],
                    "top_phrases": profile["top_phrases"],
                    "common_endings": profile["common_endings"],
                    "length_hint": profile["length_hint"],
                    "punctuation_style": profile["punctuation_style"],
                },
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "上传的文件不是有效的 JSON 格式"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/style/<user_id>", methods=["GET"])
def get_style(user_id: str):
    profile = style_profiles.get(user_id)
    if not profile:
        return jsonify({"error": "该用户尚未上传聊天记录"}), 404

    return jsonify(
        {
            "user_id": user_id,
            "analysis": {
                "message_count": profile["message_count"],
                "average_length": profile["average_length"],
                "tone": profile["tone"],
                "top_words": profile["top_words"],
                "top_phrases": profile["top_phrases"],
                "common_endings": profile["common_endings"],
                "length_hint": profile["length_hint"],
                "punctuation_style": profile["punctuation_style"],
            },
        }
    )


@app.route("/style/<user_id>", methods=["DELETE"])
def delete_style(user_id: str):
    if user_id in style_profiles:
        del style_profiles[user_id]
        _save_style_profiles()
    conversation_history.pop(user_id, None)
    return jsonify({"message": "已清除该用户的风格与会话记录", "user_id": user_id})


_load_style_profiles()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
