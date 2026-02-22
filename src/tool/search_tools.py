import chromadb
from langchain_core.tools import tool
from tavily import TavilyClient

from src.config.settings import settings

TAVILY_API_KEY = settings.TAVILY_API_KEY
CHROMA_DIR = settings.CHROMA_DIR


@tool("web_search", return_direct=False)
def web_search(query: str) -> str:
    """Search medical web results with Tavily and return concise snippets."""
    max_results = settings.WEB_SEARCH_MAX_RESULTS
    if not TAVILY_API_KEY:
        return "Tavily API key 未配置，跳过网络搜索。"
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query, max_results=max_results)
        output = []
        for r in results.get("results", []):
            output.append(f"- {r['title']}: {r['content'][:300]}\n  来源: {r['url']}")
        return "\n".join(output) if output else "未找到相关结果。"
    except Exception as e:
        return f"网络搜索失败: {e}"


@tool("rag_search", return_direct=False)
def rag_search(query: str) -> str:
    """Search local medical knowledge base (Chroma) and return top passages."""
    n_results = settings.RAG_TOP_K
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection("medical_knowledge")
        results = collection.query(query_texts=[query], n_results=n_results)
        if results and results["documents"] and results["documents"][0]:
            return "\n---\n".join(results["documents"][0])
    except Exception:
        pass
    return "知识库中未找到相关内容。"
