import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.db.models import Base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://nason_admin:command_local_pass@postgres:5432/command_db",
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text("ALTER TABLE detection_events ADD COLUMN IF NOT EXISTS ai_notes TEXT")
        )
        await conn.execute(
            text("ALTER TABLE detection_events ALTER COLUMN sub_type TYPE VARCHAR(120)")
        )
        await conn.execute(
            text("ALTER TABLE detection_events ALTER COLUMN attributes TYPE VARCHAR(255)")
        )
        await conn.execute(
            text("ALTER TABLE detection_events ALTER COLUMN full_label TYPE VARCHAR(160)")
        )
        await conn.execute(
            text("ALTER TABLE camera_schedules ADD COLUMN IF NOT EXISTS alert_whitelist BOOLEAN DEFAULT FALSE")
        )
        await conn.execute(
            text(
                "INSERT INTO camera_schedules (camera_id, mode, start_time, end_time, alert_whitelist) "
                "VALUES ('tapo_1', 'schedule', '23:30', '05:00', FALSE), "
                "       ('tapo_2', 'schedule', '23:30', '05:00', FALSE) "
                "ON CONFLICT (camera_id) DO NOTHING"
            )
        )
        await conn.execute(
            text(
                "UPDATE detection_events "
                "SET ai_notes = COALESCE(ai_notes, split_part(attributes, '|', 2)) "
                "WHERE attributes LIKE '%|%'"
            )
        )
        await conn.execute(
            text(
                "UPDATE detection_events "
                "SET sub_type = 'ID: ' || upper(sub_type) "
                "WHERE category = 'person' AND upper(sub_type) IN ('ISAAC', 'JENNIFER', 'BEAU', 'LYLA')"
            )
        )
