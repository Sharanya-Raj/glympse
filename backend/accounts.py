"""Local accounts, expiring sessions, and isolated user histories/predictions."""
import hashlib
import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager


class Accounts:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, username TEXT UNIQUE,
                    salt TEXT, password_hash TEXT, history TEXT, revision INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id TEXT, expires REAL);
                CREATE TABLE IF NOT EXISTS forecasts (user_id TEXT, meal_time TEXT, version TEXT,
                    payload TEXT, PRIMARY KEY(user_id, meal_time));
                CREATE TABLE IF NOT EXISTS simulations (user_id TEXT PRIMARY KEY, base_minutes REAL,
                    anchor REAL, seconds_per_reading REAL, paused INTEGER);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def password_hash(password, salt):
        return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()

    def signup(self, username, password):
        user_id, salt = uuid.uuid4().hex, secrets.token_hex(16)
        with self.connect() as db:
            db.execute("INSERT INTO users(id,username,salt,password_hash) VALUES(?,?,?,?)",
                (user_id, username.casefold(), salt, self.password_hash(password, salt)))
        return user_id

    def ensure_user(self, username, password):
        """Create or re-key a configured account; configuration is the source of truth."""
        with self.connect() as db:
            row = db.execute("SELECT id,salt,password_hash FROM users WHERE username=?", (username.casefold(),)).fetchone()
            if row and secrets.compare_digest(self.password_hash(password, row[1]), row[2]):
                return row[0]
            if row:
                salt = secrets.token_hex(16)
                db.execute("UPDATE users SET salt=?,password_hash=? WHERE id=?",
                           (salt, self.password_hash(password, salt), row[0]))
                db.execute("DELETE FROM sessions WHERE user_id=?", (row[0],))
                return row[0]
        return self.signup(username, password)

    def username(self, user_id):
        with self.connect() as db:
            row = db.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
        return row[0] if row else None

    def logout(self, token):
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))

    def login(self, username, password):
        with self.connect() as db:
            row = db.execute("SELECT id,salt,password_hash FROM users WHERE username=?", (username.casefold(),)).fetchone()
            hashed = self.password_hash(password, row[1] if row else "00" * 16)
            if not row or not secrets.compare_digest(hashed, row[2]):
                return None
            token = secrets.token_urlsafe(32)
            db.execute("INSERT INTO sessions VALUES(?,?,?)",
                (hashlib.sha256(token.encode()).hexdigest(), row[0], time.time() + 86400))
        return token

    def authenticate(self, token):
        with self.connect() as db:
            row = db.execute("SELECT user_id FROM sessions WHERE token_hash=? AND expires>?",
                (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
        return row[0] if row else None

    def history(self, user_id):
        with self.connect() as db:
            return db.execute("SELECT history,revision FROM users WHERE id=?", (user_id,)).fetchone()

    def save_history(self, user_id, csv, revision):
        with self.connect() as db:
            changed = db.execute("UPDATE users SET history=?,revision=revision+1 WHERE id=? AND revision=?",
                                 (csv, user_id, revision)).rowcount
            if changed != 1:
                raise ValueError("Concurrent upload detected; retry")
            db.execute("DELETE FROM forecasts WHERE user_id=?", (user_id,))

    def simulation(self, user_id):
        with self.connect() as db:
            row = db.execute("SELECT base_minutes,anchor,seconds_per_reading,paused FROM simulations WHERE user_id=?",
                             (user_id,)).fetchone()
        return None if row is None else {"base_minutes": row[0], "anchor": row[1],
                                         "seconds_per_reading": row[2], "paused": bool(row[3])}

    def save_simulation(self, user_id, base_minutes, anchor, seconds_per_reading, paused):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO simulations VALUES(?,?,?,?,?)",
                       (user_id, base_minutes, anchor, seconds_per_reading, int(paused)))

    def clear_forecasts(self, user_id):
        with self.connect() as db:
            db.execute("DELETE FROM forecasts WHERE user_id=?", (user_id,))

    def save_forecast(self, user_id, payload, revision):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT revision FROM users WHERE id=?", (user_id,)).fetchone()[0]
            if current != revision:
                raise ValueError("History changed during prediction; retry")
            db.execute("INSERT OR REPLACE INTO forecasts VALUES(?,?,?,?)",
                (user_id, payload["meal_time"], payload["model_version"], json.dumps(payload)))

    def forecasts(self, user_id, version):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM forecasts WHERE user_id=? AND version=?",
                              (user_id, version)).fetchall()
        return [json.loads(row[0]) for row in rows]
