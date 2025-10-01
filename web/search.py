from duckduckgo_search import DDGS


def search_web(query: str, num=3):
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=num)]
    return "\n".join([f"{r['title']}: {r['body']}" for r in results])
