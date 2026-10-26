from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    camera_id: Mapped[str] = mapped_column(String(50), index=True)
    motion_slot: Mapped[str] = mapped_column(String(32))
    category: Mapped[str] = mapped_column(String(50))
    sub_type: Mapped[str] = mapped_column(String(50))
    attributes: Mapped[str] = mapped_column(String(100))
    full_label: Mapped[str] = mapped_column(String(150), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    bbox_x: Mapped[float] = mapped_column(Float)
    bbox_y: Mapped[float] = mapped_column(Float)
    bbox_w: Mapped[float] = mapped_column(Float)
    bbox_h: Mapped[float] = mapped_column(Float)
    ai_notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class CameraZone(Base):
    __tablename__ = "camera_zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), index=True)
    name: Mapped[str] = mapped_column(String(64))
    zone_type: Mapped[str] = mapped_column(String(32))
    points_json: Mapped[str] = mapped_column(Text)


class CameraSchedule(Base):
    __tablename__ = "camera_schedules"

    camera_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    mode: Mapped[str] = mapped_column(String(24), default="always_record")
    start_time: Mapped[str] = mapped_column(String(10), default="21:00")
    end_time: Mapped[str] = mapped_column(String(10), default="06:00")


class KnownFace(Base):
    __tablename__ = "known_faces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(24), default="whitelist")  # 'whitelist' | 'visitor' | 'watchlist'
    embedding_json: Mapped[str] = mapped_column(Text)
    sample_crop: Mapped[str | None] = mapped_column(String(160), nullable=True)
    sightings: Mapped[int] = mapped_column(Integer, default=1)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
