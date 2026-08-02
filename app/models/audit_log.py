"""Audit log ORM model."""
import uuid
from sqlalchemy import String, Text, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

AuditRoleEnum = SAEnum(
    "nurse", "doctor", "admin", "government",
    name="audit_role_enum",
)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user: Mapped[str] = mapped_column(String, nullable=False)       # display name
    role: Mapped[str] = mapped_column(AuditRoleEnum, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[str] = mapped_column(String, nullable=False)  # ISO 8601
