"""User ORM model."""
import uuid
from sqlalchemy import String, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

RoleEnum = SAEnum(
    "nurse", "doctor", "admin", "government",
    name="role_enum",
)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(RoleEnum, nullable=False)
    hospital_id: Mapped[str] = mapped_column(String, nullable=False)
    hospital_name: Mapped[str] = mapped_column(String, nullable=False)
