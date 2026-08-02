"""Transfer request ORM model."""
import uuid
from sqlalchemy import String, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

TransferStatusEnum = SAEnum("pending", "approved", "rejected", name="transfer_status_enum")


class Transfer(Base):
    __tablename__ = "transfers"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_name: Mapped[str] = mapped_column(String, nullable=False)
    source_hospital: Mapped[str] = mapped_column(String, nullable=False)
    destination_hospital: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(TransferStatusEnum, nullable=False, default="pending")
    requested_date: Mapped[str] = mapped_column(String, nullable=False)  # YYYY-MM-DD
