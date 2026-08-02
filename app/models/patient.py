"""Patient ORM model."""
import uuid
from sqlalchemy import String, Integer, Text, Enum as SAEnum, Date, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

GenderEnum = SAEnum("Male", "Female", "Other", name="gender_enum")
SeverityEnum = SAEnum("Critical", "Moderate", "Minor", name="severity_enum")
PriorityEnum = SAEnum("High", "Medium", "Low", name="priority_enum")


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(GenderEnum, nullable=False)
    admission_date: Mapped[str] = mapped_column(String, nullable=False)  # YYYY-MM-DD
    collision_type: Mapped[str] = mapped_column(String, nullable=False)
    injury_description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(SeverityEnum, nullable=False)
    priority: Mapped[str] = mapped_column(PriorityEnum, nullable=False)
    predicted_los_days: Mapped[int] = mapped_column(Integer, nullable=False)
    created_by: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    hospital_id: Mapped[str] = mapped_column(String, nullable=False)
