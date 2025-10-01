import matplotlib
import matplotlib.pyplot as plt
import io
import requests
import json
from config import OPENROUTER_API_KEY

matplotlib.use("Agg")


def ask_grok_for_plot(prompt: str) -> dict | None:
    """
    Просим Grok вернуть данные для графика в JSON.
    Формат JSON:
    {
        "type": "line" | "bar" | "pie",
        "x": [...],
        "y": [...],
        "title": "...",
        "xlabel": "...",
        "ylabel": "..."
    }
    Если график не нужен — возвращаем None
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    system_msg = (
        "Если пользователь просит график, верни JSON с полями: "
        "{\"type\": \"line|bar|pie\", \"x\": [...], \"y\": [...], "
        "\"title\": \"...\", \"xlabel\": \"...\", \"ylabel\": \"...\"}. "
        "Если график не нужен — просто дай текст."
    )
    data = {
        "model": "x-ai/grok-4-fast:free",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=30)
        j = r.json()
        answer = j["choices"][0]["message"]["content"]
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            return None
    except Exception as e:
        print("Ошибка Grok:", e)
        return None


def render_plot(data: dict):
    """
    Рисуем график по JSON.
    Поддерживаются line, bar и pie.
    """
    plot_type = data.get("type", "line")
    x = data.get("x", [])
    y = data.get("y", [])
    title = data.get("title", "График")
    xlabel = data.get("xlabel", "X")
    ylabel = data.get("ylabel", "Y")

    plt.figure(figsize=(6, 4))

    if plot_type == "line":
        plt.plot(x, y, marker="o")
    elif plot_type == "bar":
        plt.bar(x, y)
    elif plot_type == "pie":
        plt.pie(y, labels=x, autopct="%1.1f%%")
    else:
        plt.plot(x, y, marker="o")

    plt.title(title)
    if plot_type != "pie":
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf
