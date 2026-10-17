from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, func
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
