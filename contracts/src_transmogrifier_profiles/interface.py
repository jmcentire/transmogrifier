# === Model Profile Cache (src_transmogrifier_profiles) v1 ===
#  Dependencies: json, logging, datetime, pathlib, pydantic
# File-based cache for model register sensitivity profiles with pre-seeded calibration data. Tracks per-model and per-task-type accuracy metrics across different linguistic registers (direct, technical, academic, narrative, casual) to enable register-aware prompt optimization.

# Module invariants:
#   - _PRESEEDED contains pre-calibrated profiles for 'claude-opus-4', 'claude-haiku-4-5', 'gpt-4o-mini', 'gemini-2-5-flash' as of 2026-03-27
#   - _ALIASES maps versioned model IDs to canonical profile names
#   - Default register fallback is always 'direct' when accuracies list is empty
#   - Default cache directory is ~/.transmogrifier/profiles
#   - Profile TTL default is 720 hours (30 days)
#   - Calibration version default is '1.0'
#   - Sample size default is 16
#   - Register spread threshold for invariance is 2.0 percentage points

class RegisterAccuracy:
    """Register-specific accuracy measurement for a model"""
    register: str                            # required, Register name (e.g., 'direct', 'technical', 'academic', 'narrative', 'casual')
    accuracy: float                          # required, Accuracy score in range [0.0, 1.0]
    sample_size: int = 16                    # optional, Number of test cases used for calibration
    task_type: str = ""                      # optional, Task type filter; empty string means aggregate across all tasks

class TaskRegisterProfile:
    """Per-task-type register accuracy breakdown"""
    task_type: str                           # required, Task category (e.g., 'factual', 'reasoning', 'code', 'analysis')
    accuracies: list[RegisterAccuracy]       # required, Accuracy measurements for each register on this task type

class ModelProfile:
    """Complete calibration profile for a single model including aggregate and per-task register sensitivities"""
    model_name: str                          # required, Canonical model identifier
    model_version: str = ""                  # optional, Model version/snapshot identifier
    provider: str = ""                       # optional, Provider name (e.g., 'anthropic', 'openai', 'gemini')
    accuracies: list[RegisterAccuracy]       # required, Aggregate accuracy across all tasks (backward compatible)
    by_task: list[TaskRegisterProfile] = []  # optional, Per-task-type accuracy breakdown
    calibrated_at: str = ""                  # optional, ISO 8601 timestamp of calibration
    ttl_hours: int = 720                     # optional, Time-to-live in hours before profile is considered expired
    calibration_version: str = "1.0"         # optional, Schema version of calibration data

class ProfileCache:
    """File-based cache manager for model profiles with memory layer and pre-seeded fallback"""
    _cache_dir: Path                         # required, Directory for persistent JSON profile storage
    _memory: dict[str, ModelProfile]         # required, In-memory cache layer keyed by canonical model name

def TaskRegisterProfile.best_register(
    self: TaskRegisterProfile,
) -> str:
    """
    Returns the register with highest accuracy for this task type. Defaults to 'direct' if no accuracies available.

    Postconditions:
      - Returns 'direct' if self.accuracies is empty
      - Otherwise returns the register field of the RegisterAccuracy with maximum accuracy

    Side effects: none
    Idempotent: no
    """
    ...

def TaskRegisterProfile.spread_pp(
    self: TaskRegisterProfile,
) -> float:
    """
    Calculates percentage point spread (max - min accuracy) * 100 for this task type. Returns 0.0 if no accuracies.

    Postconditions:
      - Returns 0.0 if self.accuracies is empty
      - Otherwise returns (max(accuracies) - min(accuracies)) * 100

    Side effects: none
    Idempotent: no
    """
    ...

def ModelProfile.spread_pp(
    self: ModelProfile,
) -> float:
    """
    Calculates aggregate percentage point spread (max - min accuracy) * 100 across all registers. Returns 0.0 if no accuracies. Computed field.

    Postconditions:
      - Returns 0.0 if self.accuracies is empty
      - Otherwise returns (max(accuracies) - min(accuracies)) * 100

    Side effects: none
    Idempotent: no
    """
    ...

def ModelProfile.is_invariant(
    self: ModelProfile,
) -> bool:
    """
    Determines if model is register-invariant (spread < 2.0 percentage points). Computed field.

    Postconditions:
      - Returns True if self.spread_pp < 2.0, False otherwise

    Side effects: none
    Idempotent: no
    """
    ...

def ModelProfile.best_register(
    self: ModelProfile,
) -> str:
    """
    Returns the aggregate register with highest accuracy. Defaults to 'direct' if no accuracies. Computed field.

    Postconditions:
      - Returns 'direct' if self.accuracies is empty
      - Otherwise returns the register field of the RegisterAccuracy with maximum accuracy

    Side effects: none
    Idempotent: no
    """
    ...

def ModelProfile.worst_register(
    self: ModelProfile,
) -> str:
    """
    Returns the aggregate register with lowest accuracy. Defaults to 'direct' if no accuracies. Computed field.

    Postconditions:
      - Returns 'direct' if self.accuracies is empty
      - Otherwise returns the register field of the RegisterAccuracy with minimum accuracy

    Side effects: none
    Idempotent: no
    """
    ...

def ModelProfile.best_register_for_task(
    self: ModelProfile,
    task_type: str,
) -> str:
    """
    Returns optimal register for specific task type. Falls back to aggregate best_register if task type not found in by_task.

    Postconditions:
      - If task_type matches a TaskRegisterProfile in self.by_task, returns that profile's best_register
      - Otherwise returns self.best_register (aggregate)

    Side effects: none
    Idempotent: no
    """
    ...

def ModelProfile.spread_for_task(
    self: ModelProfile,
    task_type: str,
) -> float:
    """
    Returns register spread for specific task type. Falls back to aggregate spread if task type not found.

    Postconditions:
      - If task_type matches a TaskRegisterProfile in self.by_task, returns that profile's spread_pp
      - Otherwise returns self.spread_pp (aggregate)

    Side effects: none
    Idempotent: no
    """
    ...

def ModelProfile.is_expired(
    self: ModelProfile,
) -> bool:
    """
    Checks if profile has exceeded its TTL based on calibrated_at timestamp. Returns False if calibrated_at is empty or parsing fails.

    Postconditions:
      - Returns False if self.calibrated_at is empty
      - Returns False if datetime parsing raises ValueError or TypeError
      - Otherwise returns True if (now - calibrated_at) > self.ttl_hours, False otherwise

    Side effects: Calls datetime.now(timezone.utc) to get current time
    Idempotent: no
    """
    ...

def ProfileCache.__init__(
    self: ProfileCache,
    cache_dir: Path | None = None,
) -> None:
    """
    Initializes ProfileCache with optional cache directory. Defaults to ~/.transmogrifier/profiles if not provided.

    Postconditions:
      - self._cache_dir is set to cache_dir if provided, otherwise Path.home() / '.transmogrifier' / 'profiles'
      - self._memory is initialized to empty dict

    Side effects: none
    Idempotent: no
    """
    ...

def ProfileCache.get(
    self: ProfileCache,
    model_name: str,
) -> ModelProfile | None:
    """
    Retrieves ModelProfile by name with alias resolution, memory cache, file cache, and pre-seeded fallback. Returns None if not found.

    Postconditions:
      - Resolves model_name via _ALIASES to canonical name
      - Returns cached unexpired profile from memory if exists
      - Otherwise loads from file, caches in memory, and returns if unexpired
      - Otherwise returns from _PRESEEDED dict if exact match exists
      - Otherwise returns from _PRESEEDED dict if partial match exists (key in canonical or canonical in key)
      - Returns None if no match found anywhere

    Side effects: May update self._memory cache, Reads from filesystem via _load_file
    Idempotent: no
    """
    ...

def ProfileCache.put(
    self: ProfileCache,
    profile: ModelProfile,
) -> Path:
    """
    Persists ModelProfile to disk and updates memory cache. Creates cache directory if it doesn't exist.

    Postconditions:
      - Creates self._cache_dir with parents if it doesn't exist
      - Writes profile JSON to {cache_dir}/{profile.model_name}.json
      - Updates self._memory[profile.model_name] with profile
      - Returns Path to written file

    Side effects: Creates directories, Writes to filesystem
    Idempotent: no
    """
    ...

def ProfileCache.invalidate(
    self: ProfileCache,
    model_name: str,
) -> bool:
    """
    Removes profile from memory cache and deletes file from disk. Returns True if file was deleted, False otherwise.

    Postconditions:
      - Resolves model_name via _ALIASES to canonical name
      - Removes canonical name from self._memory if present
      - Deletes {cache_dir}/{canonical}.json if it exists and returns True
      - Returns False if file does not exist

    Side effects: Removes from memory cache, May delete file from filesystem
    Idempotent: no
    """
    ...

def ProfileCache.list_profiles(
    self: ProfileCache,
) -> list[ModelProfile]:
    """
    Returns list of all available profiles from pre-seeded data and cached files, deduplicating by model_name.

    Postconditions:
      - Returns list starting with all _PRESEEDED profiles
      - Appends profiles from {cache_dir}/*.json files that don't have matching model_name in pre-seeded
      - Silently skips files that fail to parse

    Side effects: Reads all .json files from cache_dir if directory exists
    Idempotent: no
    """
    ...

def ProfileCache._load_file(
    self: ProfileCache,
    model_name: str,
) -> ModelProfile | None:
    """
    Internal method to load ModelProfile from JSON file. Returns None if file doesn't exist or parsing fails.

    Postconditions:
      - Returns None if {cache_dir}/{model_name}.json does not exist
      - Returns ModelProfile instance if file exists and parses successfully
      - Returns None and logs debug message if parsing fails

    Side effects: Reads from filesystem, Logs debug message on parse failure
    Idempotent: no
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['RegisterAccuracy', 'TaskRegisterProfile', 'ModelProfile', 'ProfileCache']
