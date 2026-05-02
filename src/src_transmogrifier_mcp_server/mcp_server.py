"""FastMCP server for Claude Code integration. Phase 9 stub."""

from __future__ import annotations


def main():
    """MCP server entry point."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        import sys
        print("MCP server requires: pip install 'transmogrifier[mcp]'", file=sys.stderr)
        sys.exit(1)

    mcp = FastMCP(
        "transmogrifier",
        instructions=(
            "Transmogrifier normalizes linguistic register for better LLM output. "
            "Use transmog_translate before sending prompts to register-sensitive models. "
            "Use transmog_detect to classify input register."
        ),
    )

    from .core import Transmogrifier, TranslationConfig, Register

    _t = Transmogrifier()

    @mcp.tool()
    def transmog_translate(text: str, model: str = "", target_register: str = "") -> dict:
        """Translate text to optimal register for the target model."""
        config = TranslationConfig()
        if target_register:
            config.target_register = Register(target_register)
        result = _t.translate(text, model=model, config=config)
        return result.model_dump()

    @mcp.tool()
    def transmog_detect(text: str) -> dict:
        """Detect the register of input text."""
        reg, conf = _t._detector.detect(text)
        return {"register": reg.value, "confidence": conf}

    @mcp.tool()
    def transmog_profiles() -> list[dict]:
        """List all cached model register sensitivity profiles."""
        return [p.model_dump() for p in _t._profile_cache.list_profiles()]

    mcp.run(transport="stdio")
