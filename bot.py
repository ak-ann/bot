import re
import os
import requests
import telebot
import logging
from dotenv import load_dotenv
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
import random


from rag.loader import search_docs
from web.search import search_web
from viz.plotter import ask_grok_for_plot, render_plot


load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "x-ai/grok-4-fast:free"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


PLOT_KEYWORDS = ["график", "диаграмма", "chart", "plot", "построй"]
WEB_KEYWORDS = ["новости", "сегодня", "актуальный", "сейчас", "2025"]


if not BOT_TOKEN or not OPENROUTER_API_KEY:
    raise ValueError(
        "BOT_TOKEN и OPENROUTER_API_KEY должны быть указаны в .env"
    )

bot = telebot.TeleBot(BOT_TOKEN)


BUTTON_SUGGESTIONS = [
    "Какой сегодня курс доллара?",
    "Последние новости о ИИ",
    "Какая погода в Москве?",
    "Главные события в мире за сегодня",
    "Что нового в технологиях?",
    "Построй график акций Tesla за год",
    "Диаграмма цен на нефть за месяц",
    "Сравни курсы EUR и USD за неделю",
    "Что такое Large Language Model?",
    "Краткая история интернета",
    "Расскажи интересный факт о космосе",
    "Какие вклады есть в Сбере?",
    "Условия по ипотеке для молодой семьи",
    "Как открыть счет в СберБанке?",
    "Расскажи историю создания Сбера",
    "Какие документы нужны для кредита?",
    "Что такое СберПрайм?",
    "Как работает программа 'Спасибо'?",
    "Последние новости о СберБанке",
    "Акции Сбера: текущая цена",
    "Построй график акций Сбера за год",
    "Диаграмма изменения ключевой ставки ЦБ",
]


def format_grok_response(text: str) -> str:
    """Адаптирует Markdown от Grok для parse_mode='MarkdownV2' в Telegram."""
    text = re.sub(r'###\s*(.*)', r'*\1*', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    text = re.sub(r'^\s*\*\s', '• ', text, flags=re.MULTILINE)

    escape_chars = r'[_`\[\]()~>#+-=|{}.!]'
    text = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)
    text = text.replace('\\*', '*').replace('\\•', '•')

    return text


def send_long_message(chat_id, text, **kwargs):
    """Отправляет длинные сообщения, разбивая их на части."""
    MAX_LENGTH = 4096
    if len(text) <= MAX_LENGTH:
        bot.send_message(chat_id, text, **kwargs)
    else:
        parts = []
        while len(text) > 0:
            if len(text) > MAX_LENGTH:
                part = text[:MAX_LENGTH]
                last_newline = part.rfind('\n')
                if last_newline != -1:
                    parts.append(part[:last_newline])
                    text = text[last_newline+1:]
                else:
                    parts.append(part)
                    text = text[MAX_LENGTH:]
            else:
                parts.append(text)
                break

        for part in parts:
            bot.send_message(chat_id, part, **kwargs)


def ask_grok(messages: list) -> str:
    """
    Отправляет запрос к модели ИИ и обрабатывает возможные ошибки.
    Принимает список сообщений в формате OpenAI.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1024
    }
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=data,
            timeout=45
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка сети при запросе к OpenRouter: {e}")
        return (
            "Простите, возникла проблема с доступом к сети."
            "Попробуйте позже."
        )
    except KeyError:
        logger.error(f"Неожиданный формат ответа от API: {response.text}")
        return "Получен странный ответ от ИИ, не могу его обработать."
    except Exception as e:
        logger.error(f"Неизвестная ошибка при запросе к Grok: {e}")
        return "Простите, что-то пошло не так. Я уже разбираюсь в проблеме."


@bot.message_handler(commands=["start", "help"])
def start(message):
    markup = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)

    try:
        selected_questions = random.sample(BUTTON_SUGGESTIONS, 3)
    except ValueError:
        selected_questions = BUTTON_SUGGESTIONS

    buttons = [KeyboardButton(text=q) for q in selected_questions]
    markup.add(*buttons)

    help_text = (
        "Здравствуйте! Я — ваш персональный ассистент.\n\n"
        "Задайте свой вопрос или воспользуйтесь одной из подсказок ниже 👇"
    )
    bot.send_message(message.chat.id, help_text, reply_markup=markup)


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    query = message.text
    chat_id = message.chat.id

    bot.send_chat_action(chat_id, 'typing')

    try:

        plot_keywords = ["график", "диаграмма", "chart", "plot", "построй"]
        if any(k.lower() in query.lower() for k in plot_keywords):
            plot_data = ask_grok_for_plot(query)
            if plot_data:
                buf = render_plot(plot_data)
                bot.send_photo(chat_id, buf)
                buf.close()
                return

        docs_context = "\n".join(search_docs(query))
        web_context = (
            search_web(query)
            if any(w in query.lower() for w in ["новости", "сегодня", "курс"])
            else ""
        )

        system_prompt = (
            "Ты — дружелюбный и полезный ИИ-ассистент."
            "Отвечай структурированно, используй Markdown для форматирования."
        )
        prompt = (
            f"Вопрос: {query}\n\n"
            f"Контекст из документов:\n{docs_context}\n\n"
            f"Контекст из интернета:\n{web_context}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        final_answer = ask_grok(messages)

        formatted_answer = format_grok_response(final_answer)
        send_long_message(chat_id, formatted_answer, parse_mode='MarkdownV2')

    except Exception as e:
        logger.error(f"Ошибка в handle_message: {e}", exc_info=True)
        bot.send_message(chat_id, "Ой, произошла ошибка. Попробуйте еще раз.")


if __name__ == '__main__':
    logger.info("🤖 Бот запущен...")
    bot.polling(none_stop=True)
