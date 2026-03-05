import pytest

from skullite import DbID, Skullite, SkulliteError, SkulliteFunction, SkulliteLogicError
from skullite.skullite import (
    _safe_id,
    _SkulliteConnection,
    _SkullitPersistentConnection,
)

SCHEMA = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"


# --- _safe_id ---


class TestSafeId:
    def test_valid(self):
        assert _safe_id("users") == '"users"'
        assert _safe_id("_col") == '"_col"'
        assert _safe_id("table1") == '"table1"'

    def test_invalid(self):
        with pytest.raises(SkulliteError, match="Invalid SQL identifier"):
            _safe_id("1bad")
        with pytest.raises(SkulliteError, match="Invalid SQL identifier"):
            _safe_id("drop; --")

    def test_star_rejected_by_default(self):
        with pytest.raises(SkulliteError, match="Invalid SQL identifier"):
            _safe_id("*")

    def test_star_allowed(self):
        assert _safe_id("*", allow_star=True) == "*"

    def test_allow_star_still_validates(self):
        assert _safe_id("col", allow_star=True) == '"col"'


# --- DbID ---


class TestDbID:
    def test_zero_is_truthy(self):
        assert bool(DbID(0)) is True

    def test_nonzero_is_truthy(self):
        assert bool(DbID(42)) is True

    def test_is_int(self):
        assert DbID(5) == 5
        assert isinstance(DbID(5), int)


# --- Skullite in-memory ---


class TestSkulliteInMemory:
    def test_default_is_persistent(self):
        db = Skullite()
        assert db.is_persistent()

    def test_empty_string_is_in_memory(self):
        db = Skullite("")
        assert db.in_memory()
        assert db.is_persistent()

    def test_init_with_script(self):
        db = Skullite(script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        assert len(db.query_all("SELECT * FROM users")) == 1

    def test_init_with_script_path(self, tmp_path):
        p = tmp_path / "schema.sql"
        p.write_text(SCHEMA)
        db = Skullite(script_path=str(p))
        db.insert("users", name="Alice", age=30)
        assert len(db.query_all("SELECT * FROM users")) == 1

    def test_script_and_script_path_exclusive(self, tmp_path):
        p = tmp_path / "schema.sql"
        p.write_text(SCHEMA)
        with pytest.raises(SkulliteError, match="mutually exclusive"):
            Skullite(script_path=str(p), script=SCHEMA)

    def test_context_manager_stays_persistent(self):
        db = Skullite(script=SCHEMA)
        with db:
            assert db.is_persistent()
            db.insert("users", name="Alice", age=30)
        assert db.is_persistent()

    def test_insert_returns_dbid(self):
        db = Skullite(script=SCHEMA)
        rid = db.insert("users", name="Alice", age=30)
        assert isinstance(rid, DbID)

    def test_insert_or_ignore_duplicate(self):
        db = Skullite(
            script="CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT UNIQUE);"
        )
        db.insert("t", val="a")
        db.insert_or_ignore("t", val="a")
        assert db.count("t", "*", "1=1") == 1

    def test_modify_many(self):
        db = Skullite(script=SCHEMA)
        db.modify_many(
            "INSERT INTO users (name, age) VALUES (?, ?)", [("Alice", 30), ("Bob", 25)]
        )
        assert len(db.query_all("SELECT * FROM users")) == 2

    def test_query_generator(self):
        db = Skullite(script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        db.insert("users", name="Bob", age=25)
        rows = list(db.query("SELECT * FROM users ORDER BY name"))
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"

    def test_query_one(self):
        db = Skullite(script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        row = db.query_one("SELECT * FROM users WHERE name = ?", ("Alice",))
        assert row["age"] == 30

    def test_query_one_no_result(self):
        db = Skullite(script=SCHEMA)
        assert db.query_one("SELECT * FROM users WHERE name = ?", ("Nobody",)) is None

    def test_select_id_found(self):
        db = Skullite(script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        uid = db.select_id("users", "id", "name = ?", ("Alice",))
        assert isinstance(uid, DbID)

    def test_select_id_not_found(self):
        db = Skullite(script=SCHEMA)
        assert db.select_id("users", "id", "name = ?", ("Nobody",)) is None

    def test_select_id_multiple_raises(self):
        db = Skullite(script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        db.insert("users", name="Alice", age=31)
        with pytest.raises(RuntimeError, match="Found 2 entries"):
            db.select_id("users", "id", "name = ?", ("Alice",))

    def test_select_id_none_in_params(self):
        db = Skullite(script=SCHEMA)
        with pytest.raises(SkulliteError, match="None not allowed"):
            db.select_id("users", "id", "name = ?", (None,))

    def test_select_id_from_values(self):
        db = Skullite(script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        uid = db.select_id_from_values("users", "id", name="Alice")
        assert isinstance(uid, DbID)

    def test_select_id_from_values_not_found(self):
        db = Skullite(script=SCHEMA)
        assert db.select_id_from_values("users", "id", name="Nobody") is None

    def test_select_id_from_values_with_none(self):
        db = Skullite(script="CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);")
        db.modify("INSERT INTO t (val) VALUES (NULL)")
        uid = db.select_id_from_values("t", "id", val=None)
        assert isinstance(uid, DbID)

    def test_select_id_from_values_multiple_raises(self):
        db = Skullite(script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        db.insert("users", name="Alice", age=31)
        with pytest.raises(RuntimeError, match="Found 2 entries"):
            db.select_id_from_values("users", "id", name="Alice")

    def test_count_star(self):
        db = Skullite(script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        db.insert("users", name="Bob", age=25)
        assert db.count("users", "*", "1=1") == 2
        assert db.count("users", "*", "age > ?", (27,)) == 1

    def test_count_with_column(self):
        db = Skullite(script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        assert db.count("users", "name", "1=1") == 1

    def test_count_none_in_params(self):
        db = Skullite(script=SCHEMA)
        with pytest.raises(SkulliteError, match="None not allowed"):
            db.count("users", "*", "name = ?", (None,))

    def test_count_from_values(self):
        db = Skullite(script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        db.insert("users", name="Bob", age=30)
        assert db.count_from_values("users", "*", age=30) == 2

    def test_count_from_values_with_none(self):
        db = Skullite(script="CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);")
        db.modify("INSERT INTO t (val) VALUES (NULL)")
        db.modify("INSERT INTO t (val) VALUES ('a')")
        assert db.count_from_values("t", "*", val=None) == 1

    def test_copy_from(self):
        src = Skullite(script=SCHEMA)
        src.insert("users", name="Alice", age=30)
        dst = Skullite()
        dst.copy_from(src)
        rows = dst.query_all("SELECT * FROM users")
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"


# --- Skullite file-based ---


class TestSkulliteFileBased:
    def test_not_persistent_by_default(self, tmp_path):
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA)
        assert not db.is_persistent()
        assert not db.in_memory()

    def test_context_manager(self, tmp_path):
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA)
        with db:
            assert db.is_persistent()
            db.insert("users", name="Alice", age=30)
        assert not db.is_persistent()

    def test_persistent_flag(self, tmp_path):
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA, persistent=True)
        assert db.is_persistent()
        db.insert("users", name="Alice", age=30)
        assert len(db.query_all("SELECT * FROM users")) == 1

    def test_persistent_context_stays_persistent(self, tmp_path):
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA, persistent=True)
        with db:
            db.insert("users", name="Alice", age=30)
        assert db.is_persistent()

    def test_query_without_persistent_raises(self, tmp_path):
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA)
        with pytest.raises(SkulliteError, match="Persistent connection required"):
            db.query("SELECT * FROM users")

    def test_operations_without_context(self, tmp_path):
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        rows = db.query_all("SELECT * FROM users")
        assert len(rows) == 1


# --- Skullite logic errors ---


class TestSkulliteLogicErrors:
    def test_enter_in_memory_no_persistent(self):
        db = Skullite()
        db._persistent = None
        with pytest.raises(SkulliteLogicError, match="Persistent expected"):
            db.__enter__()

    def test_enter_on_disk_already_persistent(self, tmp_path):
        db = Skullite(str(tmp_path / "test.db"))
        db._persistent = _SkullitPersistentConnection(None)
        with pytest.raises(SkulliteLogicError, match="Persistent not expected"):
            db.__enter__()

    def test_exit_no_persistent(self):
        db = Skullite()
        db._persistent = None
        with pytest.raises(
            SkulliteLogicError, match="Persistent expected when exiting"
        ):
            db.__exit__(None, None, None)


# --- SQL injection protection ---


class TestSqlInjection:
    def test_insert_invalid_table(self):
        db = Skullite(script=SCHEMA)
        with pytest.raises(SkulliteError, match="Invalid SQL identifier"):
            db.insert("users; DROP TABLE users", name="Alice", age=30)

    def test_insert_invalid_column(self):
        db = Skullite(script=SCHEMA)
        with pytest.raises(SkulliteError, match="Invalid SQL identifier"):
            db.insert("users", **{"na me": "Alice"})

    def test_insert_or_ignore_invalid_table(self):
        db = Skullite(script=SCHEMA)
        with pytest.raises(SkulliteError, match="Invalid SQL identifier"):
            db.insert_or_ignore("bad table", name="a")

    def test_select_id_invalid_table(self):
        db = Skullite(script=SCHEMA)
        with pytest.raises(SkulliteError, match="Invalid SQL identifier"):
            db.select_id("bad table", "id", "1=1")

    def test_count_invalid_table(self):
        db = Skullite(script=SCHEMA)
        with pytest.raises(SkulliteError, match="Invalid SQL identifier"):
            db.count("bad table", "*", "1=1")


# --- Custom functions ---


class TestCustomFunctions:
    def test_custom_function(self):
        fn = SkulliteFunction(function=lambda x: x.upper(), name="my_upper", nb_args=1)
        db = Skullite(
            script="CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);", functions=[fn]
        )
        db.insert("t", val="hello")
        row = db.query_one("SELECT my_upper(val) as v FROM t")
        assert row["v"] == "HELLO"

    def test_non_deterministic_function(self):
        import random

        fn = SkulliteFunction(
            function=lambda: random.randint(1, 100),
            name="my_rand",
            nb_args=0,
            deterministic=False,
        )
        db = Skullite(script="CREATE TABLE t (id INTEGER PRIMARY KEY);", functions=[fn])
        db.modify("INSERT INTO t (id) VALUES (1)")
        row = db.query_one("SELECT my_rand() as r FROM t")
        assert isinstance(row["r"], int)


# --- _SkulliteConnection direct tests ---


class TestSkulliteConnectionDirect:
    def test_context_manager_closes(self):
        with _SkulliteConnection(None) as conn:
            conn.cursor.execute("SELECT 1")
            assert conn.cursor.fetchone()[0] == 1

    def test_close(self):
        conn = _SkulliteConnection(None)
        conn.close()

    def test_script(self):
        conn = _SkulliteConnection(None)
        conn.script(SCHEMA)
        conn.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        assert len(conn.cursor.fetchall()) == 1
        conn.close()

    def test_modify_execute(self):
        conn = _SkulliteConnection(None)
        conn.script(SCHEMA)
        result = conn.modify(
            "INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 30)
        )
        assert isinstance(result, DbID)
        conn.close()

    def test_modify_executemany(self):
        conn = _SkulliteConnection(None)
        conn.script(SCHEMA)
        conn.modify(
            "INSERT INTO users (name, age) VALUES (?, ?)",
            [("Alice", 30), ("Bob", 25)],
            many=True,
        )
        conn.close()

    def test_query_debug(self, capsys):
        conn = _SkulliteConnection(None, debug=True)
        conn.script(SCHEMA)
        conn.modify("INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 30))
        list(conn.query("SELECT * FROM users"))
        out = capsys.readouterr().out
        assert "[query]" in out
        assert "[params]" in out
        conn.close()

    def test_query_error(self, capsys):
        conn = _SkulliteConnection(None)
        gen = conn.query("SELECT * FROM nonexistent")
        with pytest.raises(Exception):
            list(gen)
        err = capsys.readouterr().err
        assert "[error]" in err
        assert "[query]" in err
        assert "[params]" in err
        conn.close()

    def test_query_all_debug(self, capsys):
        conn = _SkulliteConnection(None, debug=True)
        conn.script(SCHEMA)
        conn.query_all("SELECT * FROM users")
        out = capsys.readouterr().out
        assert "[query]" in out
        conn.close()

    def test_query_one(self):
        conn = _SkulliteConnection(None)
        conn.script(SCHEMA)
        conn.modify("INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 30))
        row = conn.query_one("SELECT * FROM users")
        assert row["name"] == "Alice"
        conn.close()

    def test_query_all_no_debug(self):
        conn = _SkulliteConnection(None, debug=False)
        conn.script(SCHEMA)
        rows = conn.query_all("SELECT * FROM users")
        assert len(rows) == 0
        conn.close()

    def test_query_no_debug(self, capsys):
        conn = _SkulliteConnection(None, debug=False)
        conn.script(SCHEMA)
        conn.modify("INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 30))
        list(conn.query("SELECT * FROM users"))
        out = capsys.readouterr().out
        assert "[query]" not in out
        conn.close()


# --- _SkullitPersistentConnection ---


class TestPersistentConnection:
    def test_exit_does_nothing(self):
        conn = _SkullitPersistentConnection(None)
        conn.__exit__(None, None, None)
        conn.cursor.execute("SELECT 1")
        assert conn.cursor.fetchone()[0] == 1
        conn.close()
