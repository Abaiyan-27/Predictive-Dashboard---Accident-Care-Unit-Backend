"""Auth Pydantic schemas."""
from pydantic import BaseModel, EmailStr
from app.schemas.user import UserOut


class LoginCredentials(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    user: UserOut
