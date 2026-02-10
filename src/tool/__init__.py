"""Tool exports and registry."""

from src.tool.image_explain import image_explain
from src.tool.web_search import web_search

tools = [web_search, image_explain]
tools_by_name = {tool.name: tool for tool in tools}

__all__ = ["web_search", "image_explain", "tools", "tools_by_name"]
