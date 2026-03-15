# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Skullite is a Python package providing a high-level helper class around SQLite (`sqlite3`). It simplifies database operations (insert, query, count, select ID) and manages connection lifecycles including persistent and in-memory modes. Zero dependencies.

## Development Commands

```bash
# Install dependencies (uses uv)
uv sync

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Build
uv build

# Tests (100% coverage on skullite/)
uv run pytest tests/ --cov=skullite --cov-report=term-missing

# Run a single test file
uv run pytest tests/test_skullite.py

# Run a single test by name
uv run pytest tests/test_skullite.py -k "test_name"
```

## Architecture

The entire library lives in a single module `skullite/skullite.py` with five public exports (`skullite/__init__.py`):

- **`Skullite`** — Main public API. Wraps connection management and exposes `modify`, `query`, `query_one`, `query_all`, `insert`, `insert_or_ignore`, `select_id`, `count`, `copy_from`, etc. Supports context manager (`with`) for persistent connections.
- **`_SkulliteConnection`** — Internal class managing a single `sqlite3` connection + cursor. Each instance opens its own connection and closes it on exit.
- **`_SkullitPersistentConnection`** — Subclass of `_SkulliteConnection` that does NOT close on `__exit__`, requiring explicit `close()`. Used for in-memory DBs, `persistent=True` mode, and `with Skullite(...)` blocks.
- **`SkulliteFunction`** — Dataclass to register custom SQLite functions on connections.
- **`DbID`** — `int` subclass whose `__bool__` always returns `True`, so a row ID of 0 is still truthy (only `None` means "no ID").
- **`SkulliteError`** / **`SkulliteLogicError`** — Custom exceptions. `SkulliteError` for user-facing errors (invalid identifiers, missing persistent connection, etc.). `SkulliteLogicError` (subclass) for internal invariant violations.

### Connection lifecycle

- In-memory DB (`db_path=None`) or `persistent=True`: a persistent connection is created at init and never auto-closed.
- File-based DB without `persistent`: connections are ephemeral (opened/closed per operation), unless the `Skullite` object is used as a context manager, which creates a temporary persistent connection.
- `query()` (generator) requires a persistent connection because the cursor must stay open during iteration.

### Thread safety

Persistent connections are stored in `threading.local()`, so each thread gets its own independent SQLite connection. Ephemeral connections (default for file-based DBs) are inherently thread-safe since each operation creates its own connection. Foreign keys are enabled via `PRAGMA foreign_keys=ON` on every new connection.

## Code Style

- Ruff for linting and formatting. `skip-magic-trailing-comma = true` in ruff format config.
- `__init__.py` ignores `F401` (unused imports) since it re-exports symbols.
- `__slots__` used on all classes.
- Python 3.13+ required.
- pytest configured with `filterwarnings = ["ignore::ResourceWarning"]`. A `conftest.py` fixture auto-cleans connections via `gc.collect()` after each test.
