"""Auth Pydantic schemas."""
from pydantic import BaseModel, EmailStr, Field
from app.schemas.user import UserOut


class LoginCredentials(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    user: UserOut


class ChangePasswordPayload(BaseModel):
    email: EmailStr
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)
