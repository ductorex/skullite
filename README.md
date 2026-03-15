# Skullite

A simple, zero-dependency, high-level helper class for SQLite in Python.

Skullite wraps Python's built-in `sqlite3` module to reduce boilerplate for common database operations (insert, query, count, select ID) while managing connection lifecycles automatically.

**Requirements:** Python 3.13+

## Installation

```bash
pip install skullite
```

## Quick Start

```python
from skullite import Skullite

# Create an in-memory database with a schema
db = Skullite(script="CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);")

# Insert rows
db.insert("users", name="Alice", age=30)
db.insert("users", name="Bob", age=25)

# Query
rows = db.query_all("SELECT * FROM users WHERE age > ?", (20,))
for row in rows:
    print(row["name"], row["age"])
```

## How Connections Work

Understanding how Skullite manages connections is key to using it effectively.

A SQLite database can be either **in-memory** (data lives in RAM, lost when the object is destroyed) or **file-based** (data is persisted to disk). Skullite handles these two cases differently.

### In-memory databases

An in-memory database is created when no path is given. Since the data only exists through the connection, Skullite keeps a **persistent connection** open for the entire lifetime of the object. There is nothing special to do:

```python
# In-memory: connection is always open, all operations just work
db = Skullite(script="CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);")
db.insert("t", val="hello")
rows = db.query_all("SELECT * FROM t")
```

### File-based databases

A file-based database is created by passing a file path. By default, Skullite uses **ephemeral connections**: each operation (insert, query, count...) opens its own connection, does its work, and closes it immediately. This is safe and simple, but has two implications:

1. There is a small overhead per operation (opening/closing connections).
2. `query()`, which returns a lazy generator, cannot work because the connection would close before you iterate the results.

```python
db = Skullite("app.db", script_path="schema.sql")

# These work fine: each call opens and closes its own connection
db.insert("users", name="Alice", age=30)
row = db.query_one("SELECT * FROM users WHERE name = ?", ("Alice",))
total = db.count("users", "*", "1=1")
```

To keep a connection open across multiple operations, use one of these two approaches:

**Context manager** (`with` block) — opens a persistent connection for the duration of the block, then closes it automatically:

```python
with db:
    # Single connection for all operations inside the block
    db.insert("users", name="Bob", age=25)
    db.insert("users", name="Charlie", age=35)
    for row in db.query("SELECT * FROM users"):  # query() works here
        print(row["name"])
# Connection is closed when exiting the block
```

**Persistent mode** (`persistent=True`) — keeps a persistent connection open from initialization, similar to an in-memory database. Useful for long-lived database objects (e.g., a server):

```python
db = Skullite("app.db", script_path="schema.sql", persistent=True)

# Connection stays open, all operations reuse it
db.insert("users", name="Alice", age=30)
for row in db.query("SELECT * FROM users"):  # query() works without `with`
    print(row["name"])
```

### Thread safety

Skullite stores persistent connections in a `threading.local()`, so each thread automatically gets its own independent SQLite connection. This means:

- **Ephemeral mode** (file-based, default): each operation creates its own connection, so each thread naturally gets its own. This is inherently thread-safe.
- **Persistent mode** (file-based, `persistent=True` or inside a `with` block): each thread gets its own persistent connection via `threading.local()`. Multiple threads can use `with db:` concurrently on the same `Skullite` instance without conflict.
- **In-memory**: the persistent connection is created on the init thread and is only visible from that thread (SQLite enforces `check_same_thread=True` by default). In-memory databases are single-thread only.

For on-disk databases, SQLite itself handles file-level locking. Concurrent reads are always safe; concurrent writes are serialized by SQLite (consider enabling WAL mode for better write concurrency).

### Summary

| Mode | Connection behavior | Thread-safe |
|------|---------------------|-------------|
| In-memory (no `db_path`) | Always persistent. | No — single thread only. |
| File-based (default) | Ephemeral. Each operation opens/closes its own connection. | Yes — each thread gets its own connection. |
| File-based + `with db:` | Persistent for the duration of the `with` block (per thread). | Yes — each thread gets its own persistent connection. |
| File-based + `persistent=True` | Always persistent (per thread). | Yes — each thread gets its own persistent connection. |

### Checking connection state

```python
db.db_path          # The database file path, or None for in-memory
db.in_memory()      # True if the database is in-memory
db.is_persistent()  # True if a persistent connection is currently active
```

## Creating a Database

### Initialization scripts

A schema can be provided at creation time, either as a string or a file path:

```python
# Inline script
db = Skullite("app.db", script="CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")

# Script file
db = Skullite("app.db", script_path="schema.sql")

# In-memory with script
db = Skullite(script="CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);")
```

`script` and `script_path` are mutually exclusive. Providing both raises `SkulliteError`.

### Foreign keys

Foreign keys are enabled by default (`PRAGMA foreign_keys=ON` is executed on every new connection). To disable them:

```python
db = Skullite("app.db", foreign_keys=False)
```

### Custom SQLite functions

You can register custom SQL functions that will be available in all connections:

```python
from skullite import Skullite, SkulliteFunction

fn = SkulliteFunction(
    function=lambda x: x.upper(),
    name="my_upper",
    nb_args=1,
    deterministic=True,  # default
)
db = Skullite(script="CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);", functions=[fn])
db.insert("t", val="hello")
row = db.query_one("SELECT my_upper(val) as v FROM t")
print(row["v"])  # "HELLO"
```

For non-deterministic functions (e.g., random), set `deterministic=False`.

## API Reference

### Inserting Data

```python
# Insert a row, returns the new row's ID as a DbID
row_id = db.insert("users", name="Alice", age=30)

# Insert or ignore (silently skips if a constraint is violated)
db.insert_or_ignore("users", name="Alice", age=30)

# Raw modification (INSERT, UPDATE, DELETE, etc.)
db.modify("UPDATE users SET age = ? WHERE name = ?", (31, "Alice"))

# Batch insert
db.modify_many(
    "INSERT INTO users (name, age) VALUES (?, ?)",
    [("Alice", 30), ("Bob", 25), ("Charlie", 35)],
)
```

### Querying Data

```python
# Fetch all rows as a list
rows = db.query_all("SELECT * FROM users")

# Fetch one row (or None if not found)
row = db.query_one("SELECT * FROM users WHERE name = ?", ("Alice",))

# Lazy iteration (requires a persistent connection)
with db:
    for row in db.query("SELECT * FROM users ORDER BY age"):
        print(row["name"])
```

All query methods return `sqlite3.Row` objects, which support both index and key access (`row["name"]` or `row[1]`).

`query()` returns a generator and requires a persistent connection (in-memory, `persistent=True`, or inside a `with` block), because the connection must stay open while the results are being consumed.

### Finding IDs

```python
# Find a single ID with a WHERE clause
user_id = db.select_id("users", "id", "name = ?", ("Alice",))
# Returns: DbID, None if not found, raises RuntimeError if multiple found

# Find a single ID with keyword arguments
user_id = db.select_id_from_values("users", "id", name="Alice", age=30)
# Handles None values correctly: `val=None` generates `val IS NULL`
```

### Counting Rows

```python
# Count with a WHERE clause
total = db.count("users", "*", "1=1")
adults = db.count("users", "*", "age >= ?", (18,))

# Count with keyword arguments
count = db.count_from_values("users", "*", age=30)
# Also handles None: `val=None` generates `val IS NULL`
```

### Copying Databases

```python
src = Skullite("source.db", persistent=True)
dst = Skullite("backup.db", persistent=True)
dst.copy_from(src)  # Copies all tables from src to dst
```

## DbID

`DbID` is an `int` subclass returned by `insert`, `insert_or_ignore`, `select_id`, and `select_id_from_values`. Its key feature: `bool(DbID(0))` returns `True`.

This solves a common Python pitfall with database IDs:

```python
user_id = db.select_id("users", "id", "name = ?", ("Alice",))

# Without DbID: if the ID is 0, this would incorrectly take the else branch
# With DbID: only None means "not found"
if user_id:
    print(f"Found user with ID {user_id}")
else:
    print("User not found")  # Only reached when user_id is None
```

## Error Handling

Skullite defines two exception classes:

- **`SkulliteError`** — raised for user-facing errors:
  - Invalid SQL identifiers (SQL injection protection)
  - Mutually exclusive arguments (`script` + `script_path`)
  - Missing persistent connection for `query()`
  - `None` in `where_parameters` (use the `_from_values` variant instead)

- **`SkulliteLogicError`** (subclass of `SkulliteError`) — raised for internal invariant violations (unexpected state in `__enter__` / `__exit__`). These indicate a bug rather than a usage error.

```python
from skullite import SkulliteError, SkulliteLogicError
```

## SQL Injection Protection

All table and column names passed to `insert`, `insert_or_ignore`, `select_id`, `select_id_from_values`, `count`, and `count_from_values` are validated against `^[a-zA-Z_][a-zA-Z0-9_]*$` and quoted. Invalid identifiers raise `SkulliteError`.

```python
# Safe: identifiers are validated and quoted
db.insert("users", name="Alice")
# Generates: INSERT INTO "users" ("name") VALUES (?)

# Rejected: raises SkulliteError
db.insert("users; DROP TABLE users", name="Alice")
```

Note: `modify`, `query`, `query_one`, and `query_all` accept raw SQL strings. Use parameterized queries (`?` placeholders) for values to prevent injection.
