# === MCP Server (src_transmogrifier_mcp_server) v1 ===
#  Dependencies: mcp.server.fastmcp, sys, src.transmogrifier.core
# FastMCP server providing MCP tool interface for linguistic register detection and translation. Exposes Transmogrifier functionality via stdio transport for Claude Code integration.

# Module invariants:
#   - FastMCP server name is always 'transmogrifier'
#   - MCP transport is always 'stdio'
#   - Single shared Transmogrifier instance (_t) used by all tool functions
#   - Tool functions are closures over _t, only accessible after main() execution
#   - TranslationConfig() creates new instance per transmog_translate call

class TransmogTranslateResult:
    """Dictionary returned by transmog_translate, serialized from TranslationResult.model_dump()"""
    pass

class TransmogDetectResult:
    """Dictionary returned by transmog_detect containing register classification"""
    register: str                            # required, Detected register enum value as string
    confidence: float                        # required, Detection confidence score

class ProfileDict:
    """Dictionary serialized from profile.model_dump()"""
    pass

def main() -> None:
    """
    MCP server entry point. Initializes FastMCP server with transmogrifier tools and runs stdio transport. Exits with code 1 if mcp.server.fastmcp import fails.

    Postconditions:
      - FastMCP server instance created with name 'transmogrifier'
      - Three tools registered: transmog_translate, transmog_detect, transmog_profiles
      - Transmogrifier instance initialized and stored in closure
      - Server running on stdio transport (blocking call)

    Errors:
      - mcp_not_installed (SystemExit): ImportError raised when importing mcp.server.fastmcp
          exit_code: 1
          stderr_message: MCP server requires: pip install 'transmogrifier[mcp]'

    Side effects: Writes to sys.stderr on import failure, Calls sys.exit(1) on import failure, Runs blocking MCP server loop, Reads from stdin and writes to stdout via stdio transport
    Idempotent: no
    """
    ...

def transmog_translate(
    text: str,
    model: str = "",
    target_register: str = "",
) -> dict:
    """
    Translate text to optimal register for the target model. Decorated as @mcp.tool() for MCP exposure. Uses Transmogrifier.translate() and returns serialized result.

    Preconditions:
      - _t (Transmogrifier instance) must be initialized in closure

    Postconditions:
      - Returns dictionary serialized from TranslationResult via model_dump()
      - If target_register provided, config.target_register set to Register(target_register)
      - Translation performed with specified model and config

    Errors:
      - invalid_register (ValueError): target_register string not valid Register enum value
          source: Register(target_register) constructor
      - translation_failure (Exception): _t.translate() raises exception
          propagated_from: Transmogrifier.translate()

    Side effects: Calls _t.translate() which may perform LLM API calls, May access or update _t internal caches
    Idempotent: no
    """
    ...

def transmog_detect(
    text: str,
) -> dict:
    """
    Detect the register of input text. Decorated as @mcp.tool() for MCP exposure. Uses Transmogrifier._detector.detect() and returns register value and confidence.

    Preconditions:
      - _t (Transmogrifier instance) must be initialized in closure
      - _t._detector must be initialized

    Postconditions:
      - Returns dict with keys 'register' (str) and 'confidence' (float)
      - register is the .value attribute of detected Register enum
      - confidence is the numeric confidence score from detector

    Errors:
      - detector_failure (Exception): _t._detector.detect() raises exception
          propagated_from: RegisterDetector.detect()

    Side effects: Accesses _t._detector (private attribute)
    Idempotent: no
    """
    ...

def transmog_profiles() -> list[dict]:
    """
    List all cached model register sensitivity profiles. Decorated as @mcp.tool() for MCP exposure. Returns serialized profile list from _t._profile_cache.

    Preconditions:
      - _t (Transmogrifier instance) must be initialized in closure
      - _t._profile_cache must be initialized

    Postconditions:
      - Returns list of dictionaries, each serialized from profile.model_dump()
      - List contains all profiles from _t._profile_cache.list_profiles()

    Errors:
      - profile_cache_failure (Exception): _t._profile_cache.list_profiles() raises exception
          propagated_from: ProfileCache.list_profiles()

    Side effects: Accesses _t._profile_cache (private attribute)
    Idempotent: no
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['TransmogTranslateResult', 'TransmogDetectResult', 'ProfileDict', 'main', 'SystemExit', 'transmog_translate', 'transmog_detect', 'transmog_profiles']
