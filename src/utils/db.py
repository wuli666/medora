import json
import os

import aiosqlite

DB_PATH = os.getenv("DB_PATH", "./data/patients.db")


async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS medical_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                record_type TEXT,
                content TEXT,
                raw_text TEXT,
                summary TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS follow_up_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                plan_content TEXT,
                status TEXT DEFAULT 'active',
                next_visit_date TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
        """)
        await db.commit()


async def get_or_create_patient(name: str, age: int | None = None, gender: str | None = None) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM patients WHERE name = ?", (name,))
        patient = await cursor.fetchone()
        if patient:
            return dict(patient)
        await db.execute(
            "INSERT INTO patients (name, age, gender) VALUES (?, ?, ?)",
            (name, age, gender),
        )
        await db.commit()
        cursor = await db.execute("SELECT * FROM patients WHERE name = ?", (name,))
        return dict(await cursor.fetchone())


async def save_record(patient_id: int, record_type: str, content: dict, raw_text: str = "", summary: str = ""):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO medical_records (patient_id, record_type, content, raw_text, summary) VALUES (?, ?, ?, ?, ?)",
            (patient_id, record_type, json.dumps(content, ensure_ascii=False), raw_text, summary),
        )
        await db.commit()


async def get_records(patient_id: int, record_type: str | None = None) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if record_type:
            cursor = await db.execute(
                "SELECT * FROM medical_records WHERE patient_id = ? AND record_type = ? ORDER BY created_at DESC",
                (patient_id, record_type),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM medical_records WHERE patient_id = ? ORDER BY created_at DESC",
                (patient_id,),
            )
        return [dict(r) for r in await cursor.fetchall()]


async def save_follow_up_plan(patient_id: int, plan_content: dict, next_visit_date: str | None = None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO follow_up_plans (patient_id, plan_content, next_visit_date) VALUES (?, ?, ?)",
            (patient_id, json.dumps(plan_content, ensure_ascii=False), next_visit_date),
        )
        await db.commit()


async def get_follow_up_plans(patient_id: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM follow_up_plans WHERE patient_id = ? ORDER BY created_at DESC",
            (patient_id,),
        )
        return [dict(r) for r in await cursor.fetchall()]
