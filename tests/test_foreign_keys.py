import sqlite3

import pytest

from skullite import Skullite


SCHEMA = """
CREATE TABLE parent (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE child (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER NOT NULL REFERENCES parent(id) ON DELETE CASCADE,
    value TEXT NOT NULL
);
"""


@pytest.fixture()
def db_fk_on():
    return Skullite(script=SCHEMA, foreign_keys=True)


@pytest.fixture()
def db_fk_off():
    return Skullite(script=SCHEMA, foreign_keys=False)


class TestForeignKeysEnabled:
    def test_pragma_is_on(self, db_fk_on):
        with db_fk_on.connect() as conn:
            conn.cursor.execute("PRAGMA foreign_keys")
            assert conn.cursor.fetchone()[0] == 1

    def test_insert_valid_child(self, db_fk_on):
        db_fk_on.insert("parent", id=1, name="Alice")
        db_fk_on.insert("child", id=1, parent_id=1, value="x")
        rows = db_fk_on.query_all("SELECT * FROM child")
        assert len(rows) == 1

    def test_insert_orphan_rejected(self, db_fk_on):
        with pytest.raises(sqlite3.IntegrityError):
            db_fk_on.insert("child", id=1, parent_id=999, value="orphan")

    def test_cascade_delete(self, db_fk_on):
        db_fk_on.insert("parent", id=1, name="Alice")
        db_fk_on.insert("child", id=1, parent_id=1, value="a")
        db_fk_on.insert("child", id=2, parent_id=1, value="b")
        db_fk_on.modify("DELETE FROM parent WHERE id = 1")
        rows = db_fk_on.query_all("SELECT * FROM child")
        assert rows == []

    def test_cascade_delete_targets_correct_children(self, db_fk_on):
        db_fk_on.insert("parent", id=1, name="Alice")
        db_fk_on.insert("parent", id=2, name="Bob")
        db_fk_on.insert("child", id=1, parent_id=1, value="a")
        db_fk_on.insert("child", id=2, parent_id=2, value="b")
        db_fk_on.modify("DELETE FROM parent WHERE id = 1")
        rows = db_fk_on.query_all("SELECT * FROM child")
        assert len(rows) == 1
        assert rows[0]["parent_id"] == 2

    def test_update_parent_id_cascades_or_rejects(self, db_fk_on):
        db_fk_on.insert("parent", id=1, name="Alice")
        db_fk_on.insert("child", id=1, parent_id=1, value="x")
        with pytest.raises(sqlite3.IntegrityError):
            db_fk_on.modify("UPDATE child SET parent_id = 999 WHERE id = 1")


class TestForeignKeysDisabled:
    def test_pragma_is_off(self, db_fk_off):
        with db_fk_off.connect() as conn:
            conn.cursor.execute("PRAGMA foreign_keys")
            assert conn.cursor.fetchone()[0] == 0

    def test_insert_orphan_allowed(self, db_fk_off):
        db_fk_off.insert("child", id=1, parent_id=999, value="orphan")
        rows = db_fk_off.query_all("SELECT * FROM child")
        assert len(rows) == 1

    def test_no_cascade_delete(self, db_fk_off):
        db_fk_off.insert("parent", id=1, name="Alice")
        db_fk_off.insert("child", id=1, parent_id=1, value="a")
        db_fk_off.modify("DELETE FROM parent WHERE id = 1")
        rows = db_fk_off.query_all("SELECT * FROM child")
        assert len(rows) == 1


class TestForeignKeysDefault:
    def test_default_is_enabled(self):
        db = Skullite(script=SCHEMA)
        with pytest.raises(sqlite3.IntegrityError):
            db.insert("child", id=1, parent_id=999, value="orphan")

    def test_on_disk_default_is_enabled(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = Skullite(db_path, script=SCHEMA)
        with pytest.raises(sqlite3.IntegrityError):
            db.insert("child", id=1, parent_id=999, value="orphan")

    def test_on_disk_persistent_default_is_enabled(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = Skullite(db_path, script=SCHEMA, persistent=True)
        with pytest.raises(sqlite3.IntegrityError):
            db.insert("child", id=1, parent_id=999, value="orphan")


class TestForeignKeysOnDiskNonPersistent:
    """Verify FK enforcement on file-based DBs where each operation
    creates a new connection (persistent=False)."""

    def test_cascade_delete(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = Skullite(db_path, script=SCHEMA, persistent=False)
        db.insert("parent", id=1, name="Alice")
        db.insert("child", id=1, parent_id=1, value="a")
        db.modify("DELETE FROM parent WHERE id = 1")
        rows = db.query_all("SELECT * FROM child")
        assert rows == []

    def test_orphan_rejected(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = Skullite(db_path, script=SCHEMA, persistent=False)
        db.insert("parent", id=1, name="Alice")
        with pytest.raises(sqlite3.IntegrityError):
            db.insert("child", id=1, parent_id=999, value="orphan")
