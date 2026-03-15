import threading

from skullite import Skullite, SkulliteError

SCHEMA = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"


class TestThreadLocalPersistent:
    """Verify that _persistent is thread-local for on-disk non-persistent DBs."""

    def test_persistent_not_visible_from_other_thread(self, tmp_path):
        """A persistent connection opened in the main thread
        should not be visible from another thread."""
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA)
        result = {}

        with db:
            assert db.is_persistent()

            def check():
                result["is_persistent"] = db.is_persistent()

            t = threading.Thread(target=check)
            t.start()
            t.join()

        assert result["is_persistent"] is False

    def test_concurrent_with_db_contexts(self, tmp_path):
        """Two threads can each open their own `with db:` context
        simultaneously without conflict (the original bug)."""
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA)
        db.insert("users", name="Alice", age=30)
        db.insert("users", name="Bob", age=25)

        barrier = threading.Barrier(2, timeout=5)
        results = {}
        errors = {}

        def worker(name):
            try:
                with db:
                    barrier.wait()
                    rows = list(db.query("SELECT * FROM users"))
                    results[name] = len(rows)
            except Exception as exc:
                errors[name] = exc

        t1 = threading.Thread(target=worker, args=("thread1",))
        t2 = threading.Thread(target=worker, args=("thread2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"
        assert results["thread1"] == 2
        assert results["thread2"] == 2

    def test_main_thread_with_db_while_other_thread_queries(self, tmp_path):
        """Reproduces the pysaurus bug: main thread holds `with db:`,
        another thread calls get_videos (also needs `with db:`)."""
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA)
        db.insert("users", name="Alice", age=30)

        barrier = threading.Barrier(2, timeout=5)
        result = {}
        error = {}

        def other_thread():
            try:
                barrier.wait()
                with db:
                    rows = list(db.query("SELECT * FROM users"))
                    result["rows"] = len(rows)
            except Exception as exc:
                error["exc"] = exc

        t = threading.Thread(target=other_thread)
        t.start()

        with db:
            barrier.wait()
            rows = list(db.query("SELECT * FROM users"))
            assert len(rows) == 1

        t.join()

        assert not error, f"Other thread error: {error}"
        assert result["rows"] == 1

    def test_thread_query_without_context_uses_temp_connection(self, tmp_path):
        """A thread that does NOT use `with db:` can still use
        query_all / insert / modify (non-persistent operations)."""
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA)
        db.insert("users", name="Alice", age=30)

        result = {}
        error = {}

        def worker():
            try:
                rows = db.query_all("SELECT * FROM users")
                result["count"] = len(rows)
            except Exception as exc:
                error["exc"] = exc

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert not error, f"Thread error: {error}"
        assert result["count"] == 1

    def test_thread_query_without_context_while_main_in_context(self, tmp_path):
        """A thread uses query_all (no `with db:`) while the main thread
        holds `with db:`. The thread should get its own connection."""
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA)
        db.insert("users", name="Alice", age=30)

        barrier = threading.Barrier(2, timeout=5)
        result = {}
        error = {}

        def worker():
            try:
                barrier.wait()
                rows = db.query_all("SELECT * FROM users")
                result["count"] = len(rows)
            except Exception as exc:
                error["exc"] = exc

        t = threading.Thread(target=worker)
        t.start()

        with db:
            barrier.wait()
            rows = list(db.query("SELECT * FROM users"))
            assert len(rows) == 1

        t.join()

        assert not error, f"Thread error: {error}"
        assert result["count"] == 1

    def test_concurrent_writes(self, tmp_path):
        """Two threads can write to the same on-disk DB concurrently."""
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA)
        barrier = threading.Barrier(2, timeout=5)
        errors = {}

        def writer(name, age):
            try:
                barrier.wait()
                db.insert("users", name=name, age=age)
            except Exception as exc:
                errors[name] = exc

        t1 = threading.Thread(target=writer, args=("Alice", 30))
        t2 = threading.Thread(target=writer, args=("Bob", 25))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"
        rows = db.query_all("SELECT * FROM users ORDER BY name")
        assert len(rows) == 2

    def test_query_requires_persistent_per_thread(self, tmp_path):
        """query() requires `with db:` even if another thread has one open."""
        db = Skullite(str(tmp_path / "test.db"), script=SCHEMA)
        error = {}

        def worker():
            try:
                list(db.query("SELECT * FROM users"))
            except SkulliteError as exc:
                error["exc"] = exc

        with db:
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        assert "exc" in error
        assert "Persistent connection required" in str(error["exc"])


class TestInMemoryThreading:
    """In-memory DBs have a single shared persistent connection,
    created on the init thread. Other threads cannot see it."""

    def test_in_memory_persistent_not_visible_from_other_thread(self):
        db = Skullite(script=SCHEMA)
        assert db.is_persistent()

        result = {}

        def check():
            result["is_persistent"] = db.is_persistent()

        t = threading.Thread(target=check)
        t.start()
        t.join()

        assert result["is_persistent"] is False
