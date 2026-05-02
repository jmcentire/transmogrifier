# === Transmogrifier MCP Server Interface (contracts_src_transmogrifier_mcp_server_interface) v1 ===
#  Dependencies: sys, mcp.server.fastmcp, src.transmogrifier.core
# FastMCP server entrypoint that exposes Transmogrifier linguistic register normalization as MCP tools for Claude Code integration. Provides tools for translating text to optimal registers, detecting input register, and listing cached model profiles.

# Module invariants:
#   - Transmogrifier instance (_t) is initialized once per main() invocation
#   - All nested tool functions close over the same _t instance
#   - Server name is always 'transmogrifier'
#   - Transport is always 'stdio'

def main() -> None:
    """
    MCP server entry point. Initializes FastMCP server with transmogrifier tools and runs stdio transport.

    Postconditions:
      - FastMCP server is initialized with name 'transmogrifier'
      - Three MCP tools are registered: transmog_translate, transmog_detect, transmog_profiles
      - Server runs on stdio transport (blocking call)

    Errors:
      - missing_mcp_dependency (ImportError): mcp.server.fastmcp cannot be imported
          exit_code: 1
          stderr_message: MCP server requires: pip install 'transmogrifier[mcp]'

    Side effects: Prints error message to stderr if mcp.server.fastmcp is not installed, Exits process with code 1 if dependencies missing, Initializes Transmogrifier instance (may load models/profiles), Runs blocking stdio server loop
    Idempotent: no
    """
    ...

def transmog_translate(
    text: str,
    model: str = "",
    target_register: str = "",
) -> dict:
    """
    Translate text to optimal register for the target model. Nested function registered as MCP tool within main().

    Preconditions:
      - text is non-empty string (implicit)
      - target_register, if provided, must be valid Register enum value

    Postconditions:
      - Returns dict representation of TranslationResult via model_dump()
      - Result contains translated text and metadata

    Errors:
      - invalid_register (ValueError): target_register is provided but not a valid Register enum value
      - translation_failure (Exception): Transmogrifier.translate() fails (backend errors, validation errors)

    Side effects: Calls Transmogrifier.translate() which may invoke LLM backend, May cache translation results internally
    Idempotent: no
    """
    ...

def transmog_detect(
    text: str,
) -> dict:
    """
    Detect the register of input text. Nested function registered as MCP tool within main().

    Preconditions:
      - text is non-empty string (implicit)

    Postconditions:
      - Returns dict with 'register' (str) and 'confidence' (float) keys
      - register is the enum value string representation

    Errors:
      - detection_failure (Exception): _detector.detect() fails

    Side effects: Accesses _t._detector (RegisterDetector instance)
    Idempotent: no
    """
    ...

def transmog_profiles() -> list[dict]:
    """
    List all cached model register sensitivity profiles. Nested function registered as MCP tool within main().

    Postconditions:
      - Returns list of dicts, each representing a model profile via model_dump()
      - May return empty list if no profiles cached

    Errors:
      - profile_list_failure (Exception): _profile_cache.list_profiles() fails

    Side effects: Accesses _t._profile_cache to read cached profiles
    Idempotent: no
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['main', 'ImportError', 'transmog_translate', 'transmog_detect', 'transmog_profiles']
