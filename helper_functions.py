import re


def format_grok_response(text: str) -> str:
    """
    Адаптирует Markdown от Grok для parse_mode='MarkdownV2' в Telegram.
    """
    text = re.sub(r'###\s*(.*)', r'*\1*', text)

    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)

    text = re.sub(r'^\s*\*\s', '• ', text, flags=re.MULTILINE)

    escape_chars = r'[_`\[\]()~>#+-=|{}.!]'
    text = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

    text = text.replace('\\*', '*').replace('\\•', '•')

    return text
