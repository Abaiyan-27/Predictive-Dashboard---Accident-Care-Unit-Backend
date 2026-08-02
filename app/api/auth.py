"""Auth router — POST /api/auth/login"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import verify_password, create_access_token
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import LoginCredentials, AuthResponse
from app.schemas.user import UserOut

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", response_model=AuthResponse)
def login(credentials: LoginCredentials, db: Session = Depends(get_db)):
    user: User | None = (
        db.query(User).filter(User.email == credentials.email).first()
    )
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    token = create_access_token({"sub": user.id, "role": user.role})
    return AuthResponse(token=token, user=UserOut.model_validate(user))
