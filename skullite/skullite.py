"""
Helper class to use a SQLite database file.

NB: About cursors and threads
-----------------------------

SQLite does not like sharing cursors across threads.
If one wants to use a same cursor in many threads, one should:

1) Open connection with parameter check_same_thread set to False:
`sqlite3.connect(db_path, check_same_thread=False)`

This allows to avoid following error:
```
sqlite3.ProgrammingError:
SQLite objects created in a thread can only be used in that same thread.
The object was created in thread id <> and this is thread id <>.
```

2) Create a threading.Lock, share this lock with cursor across threads,
and use this lock whenever a SQL query is executed.

This was the strategy in older versions of this module.
However, this current code, we create a new cursor for each request.
So, we should not risk using a same cursor in different threads anymore.
So, in current code, using a lock is useless.
"""

import logging
import re
import sqlite3
import sys
from dataclasses import dataclass
from typing import Callable, Generator, Iterable, Self


logger = logging.getLogger(__name__)


_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _safe_id(name: str, *, allow_star: bool = False) -> str:
    """Validate and quote a SQL identifier (table or column name)."""
    if allow_star and name == "*":
        return "*"
    if not _IDENTIFIER_RE.match(name):
        raise SkulliteError(f"Invalid SQL identifier: {name!r}")
    return f'"{name}"'


class SkulliteError(Exception):
    pass


class SkulliteLogicError(SkulliteError):
    """Potential Skullite internal error."""


@dataclass(slots=True, frozen=True)
class SkulliteFunction:
    function: Callable
    name: str
    nb_args: int
    deterministic: bool = True


class DbID(int):
    """Wrapper for database ID.

    A database ID should always be evaluated to True if exists, even if its value is 0.

    If a database ID does not exist, database class will return None.

    This wrapper allows to use Python syntax (this_id or that_id) and make sure
    this_id will be returned even if this_id is 0 (we expect this_id to be None if
    related id is invalid or non-existent).
    """

    def __bool__(self):
        return True


class Skullite:
    __slots__ = (
        "_debug",
        "_db_path",
        "_functions",
        "_persistent",
        "_persistent_is_required",
    )

    def __init__(
        self,
        db_path: str | None = None,
        /,
        *,
        script_path: str | None = None,
        script: str | None = None,
        functions: Iterable[SkulliteFunction] = (),
        persistent: bool = False,
    ):
        """
        Open (or create) and populate tables (if necessary)
        in database at given path.
        """
        self._debug = False
        self._db_path = db_path or None
        self._functions = tuple(functions)
        # Used in context
        self._persistent: _SkullitPersistentConnection | None = None
        self._persistent_is_required = bool(persistent)
        # Create persistent if in memory or persistent requested
        if self._db_path is None or self._persistent_is_required:
            self._persistent = _SkullitPersistentConnection(
                self._db_path, debug=self._debug, functions=self._functions
            )
        # Execute script if given
        if script_path is not None:
            if script is not None:
                raise SkulliteError("script_path and script are mutually exclusive")
            with open(script_path, mode="r", encoding="utf-8") as script_file:
                script = script_file.read()

        if script is not None:
            with self.connect() as connection:
                connection.script(script)

    def __enter__(self) -> Self:
        """
        Create a persistent connection
        (only if not in memory and persistent not requested).
        This connection does not close automatically.
        It is instead explicitly closed when exiting this object.
        """
        logger.info("[skullite] entering persistent connection")
        if self._db_path is None or self._persistent_is_required:
            if self._persistent is None:
                raise SkulliteLogicError(
                    "Persistent expected if in-memory or explicitly required."
                )
        else:
            if self._persistent is not None:
                raise SkulliteLogicError(
                    "Persistent not expected if on-disk and not explicitly required."
                )
            self._persistent = _SkullitPersistentConnection(
                self._db_path, debug=self._debug, functions=self._functions
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("[skullite] exiting persistent connection")
        # Explicitly close persistent connection
        # NB: Only if not in memory and persistent not requested
        if self._persistent is None:
            raise SkulliteLogicError("Persistent expected when exiting a context.")
        if self._db_path is not None and not self._persistent_is_required:
            self._persistent.close()
            # Then remove persistent object
            self._persistent = None

    def in_memory(self) -> bool:
        return self._db_path is None

    def is_persistent(self) -> bool:
        return self._persistent is not None

    def _need_persistent(self):
        if not self._persistent:
            raise SkulliteError(
                "Persistent connection required. "
                "Please use with-statement on this object first."
            )

    def connect(self) -> "_SkulliteConnection":
        if self._db_path is None or self._persistent_is_required:
            self._need_persistent()
        return self._persistent or _SkulliteConnection(
            self._db_path, debug=self._debug, functions=self._functions
        )

    def modify(self, query, parameters=(), many=False) -> DbID | None:
        with self.connect() as connection:
            return connection.modify(query, parameters, many)

    def modify_many(self, query, parameters=()) -> DbID | None:
        return self.modify(query, parameters, many=True)

    def query(self, query, parameters=()) -> Generator[sqlite3.Row, None, None]:
        self._need_persistent()
        with self.connect() as connection:
            return connection.query(query, parameters)

    def query_one(self, query, parameters=()) -> sqlite3.Row:
        with self.connect() as connection:
            return connection.query_one(query, parameters)

    def query_all(self, query, parameters=()) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.query_all(query, parameters)

    def insert(self, table: str, **kwargs) -> DbID | None:
        """Insert a row in a table and return new row ID."""
        columns = list(kwargs)
        values = [kwargs[column] for column in columns]
        return self.modify(
            f"INSERT INTO {_safe_id(table)} ({', '.join(_safe_id(c) for c in columns)}) "
            f"VALUES ({', '.join('?' * len(columns))})",
            values,
        )

    def insert_or_ignore(self, table: str, **kwargs) -> DbID | None:
        """Insert a row in a table and return new row ID."""
        columns = list(kwargs)
        values = [kwargs[column] for column in columns]
        return self.modify(
            f"INSERT OR IGNORE INTO {_safe_id(table)} ({', '.join(_safe_id(c) for c in columns)}) "
            f"VALUES ({', '.join('?' * len(columns))})",
            values,
        )

    def select_id(self, table, column, where_query, where_parameters=()) -> DbID | None:
        with self.connect() as connection:
            return connection.select_id(table, column, where_query, where_parameters)

    def select_id_from_values(self, table, column, **values) -> DbID | None:
        with self.connect() as connection:
            return connection.select_id_from_values(table, column, **values)

    def count(self, table, column, where_query, where_parameters=()) -> int:
        with self.connect() as connection:
            return connection.count(table, column, where_query, where_parameters)

    def count_from_values(self, table, column, **values) -> int:
        with self.connect() as connection:
            return connection.count_from_values(table, column, **values)

    def copy_from(self, other: "Skullite"):
        """Copy all tables from another database."""
        with self.connect() as connection:
            with other.connect() as other_connection:
                other_connection.connection.backup(connection.connection)


class _SkulliteConnection:
    __slots__ = ("connection", "cursor", "debug")

    def __init__(
        self,
        db_path: str | None,
        *,
        debug=False,
        functions: tuple[SkulliteFunction, ...] = (),
    ):
        """
        Open (or create) and populate tables (if necessary)
        in database at given path.
        """
        self.debug = debug
        self.connection = sqlite3.connect(db_path or ":memory:")
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()
        self.cursor.arraysize = 1000

        for fn in functions:
            self.connection.create_function(
                fn.name, fn.nb_args, fn.function, deterministic=fn.deterministic
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

    def close(self):
        self.connection.close()

    def script(self, script: str):
        self.cursor.executescript(script)
        self.connection.commit()

    def modify(self, query, parameters=(), many=False) -> DbID | None:
        """
        Execute a modification query (INSERT, UPDATE, etc.).
        Return last inserted row ID, or None if no row was inserted.
        """
        if many:
            self.cursor.executemany(query, parameters)
        else:
            self.cursor.execute(query, parameters)
        self.connection.commit()
        last_id = self.cursor.lastrowid
        return last_id if last_id is None else DbID(last_id)

    def query(self, query, parameters=()) -> Generator[sqlite3.Row, None, None]:
        if self.debug:
            print(f"[query] {query}")
            print(f"[params] {parameters}")
        try:
            self.cursor.execute(query, parameters)
            yield from self.cursor
        except Exception as exc:
            print("[error]", type(exc), exc, file=sys.stderr)
            print(f"[query] {query}", file=sys.stderr)
            print(f"[params] {repr(parameters)}", file=sys.stderr)
            raise exc

    def query_one(self, query, parameters=()) -> sqlite3.Row:
        self.cursor.execute(query, parameters)
        return self.cursor.fetchone()

    def query_all(self, query, parameters=()) -> list[sqlite3.Row]:
        if self.debug:
            print(f"[query] {query}")
            print(f"[params] {parameters}")
        self.cursor.execute(query, parameters)
        return self.cursor.fetchall()

    def select_id(self, table, column, where_query, where_parameters=()) -> DbID | None:
        """
        Select one ID from a table and return it if found, else None.
        If more than 1 ID is found, raise a RuntimeError.
        """
        if None in where_parameters:
            raise SkulliteError(
                "None not allowed in parameters. Instead use `field=None` in select_id_from_values"
            )
        safe_table = _safe_id(table)
        safe_column = _safe_id(column)
        self.cursor.execute(
            f"SELECT {safe_column} FROM {safe_table} WHERE {where_query}",
            where_parameters,
        )
        results = self.cursor.fetchall()
        if len(results) == 0:
            return None
        elif len(results) == 1:
            return DbID(results[0][0])
        else:
            raise RuntimeError(f"Found {len(results)} entries for {table}.{column}")

    def select_id_from_values(self, table, column, **values) -> DbID | None:
        safe_table = _safe_id(table)
        safe_column = _safe_id(column)
        where_pieces = []
        where_parameters = []
        for key, value in values.items():
            safe_key = _safe_id(key)
            if value is None:
                where_pieces.append(f"{safe_key} IS NULL")
            else:
                where_pieces.append(f"{safe_key} = ?")
                where_parameters.append(value)
        where_query = " AND ".join(where_pieces)
        self.cursor.execute(
            f"SELECT {safe_column} FROM {safe_table} WHERE {where_query}",
            where_parameters,
        )
        results = self.cursor.fetchall()
        if len(results) == 0:
            return None
        elif len(results) == 1:
            return DbID(results[0][0])
        else:
            raise RuntimeError(f"Found {len(results)} entries for {table}.{column}")

    def count(self, table, column, where_query, where_parameters=()) -> int:
        """Select and return count from a table."""
        if None in where_parameters:
            raise SkulliteError(
                "None not allowed in parameters. Instead use `field=None` in count_from_values"
            )
        safe_table = _safe_id(table)
        safe_column = _safe_id(column, allow_star=True)
        self.cursor.execute(
            f"SELECT COUNT({safe_column}) FROM {safe_table} WHERE {where_query}",
            where_parameters,
        )
        return self.cursor.fetchone()[0]

    def count_from_values(self, table, column, **values) -> int:
        safe_table = _safe_id(table)
        safe_column = _safe_id(column, allow_star=True)
        where_pieces = []
        where_parameters = []
        for key, value in values.items():
            safe_key = _safe_id(key)
            if value is None:
                where_pieces.append(f"{safe_key} IS NULL")
            else:
                where_pieces.append(f"{safe_key} = ?")
                where_parameters.append(value)
        where_query = " AND ".join(where_pieces)
        self.cursor.execute(
            f"SELECT COUNT({safe_column}) FROM {safe_table} WHERE {where_query}",
            where_parameters,
        )
        return self.cursor.fetchone()[0]


class _SkullitPersistentConnection(_SkulliteConnection):
    def __exit__(self, exc_type, exc_val, exc_tb):
        """No explicit exit here. We must close manually."""
        pass
