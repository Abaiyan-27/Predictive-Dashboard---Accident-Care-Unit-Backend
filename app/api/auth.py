"""Auth router — POST /api/auth/login"""
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.core.security import create_access_token
from app.db.session import get_auth_supabase, get_supabase
from app.schemas.auth import LoginCredentials, AuthResponse
from app.schemas.user import UserOut

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", response_model=AuthResponse)
def login(
    credentials: LoginCredentials,
    auth_supabase: Client = Depends(get_auth_supabase),
    supabase: Client = Depends(get_supabase),
):
    """
    Authenticate via Supabase Auth, then look up the user in our User table
    and return a JWT + user profile.
    """
    try:
        auth_response = auth_supabase.auth.sign_in_with_password({
            "email": credentials.email,
            "password": credentials.password,
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    if not auth_response.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    # Look up user in our User table by email
    result = supabase.table("User").select("*").eq("email", credentials.email).execute()

    if result.data:
        user_data = result.data[0].copy()
        hospital_id = user_data.get("hospital_id")
        if hospital_id:
            hospital_result = (
                supabase.table("Hospital")
                .select("name")
                .eq("hospital_id", hospital_id)
                .limit(1)
                .execute()
            )
            if hospital_result.data:
                user_data["hospital_name"] = hospital_result.data[0].get("name")
        token = create_access_token({"sub": user_data["user_id"], "role": user_data["role"]})
        return AuthResponse(
            token=token,
            user=UserOut(**user_data),
        )

    # Check Government_Officer table
    result = supabase.table("Government_Officer").select("*").eq("email", credentials.email).execute()

    if result.data:
        officer_data = result.data[0]
        token = create_access_token({"sub": officer_data["officer_id"], "role": "government"})
        return AuthResponse(
            token=token,
            user=UserOut(
                user_id=officer_data["officer_id"],
                name=officer_data["name"],
                email=officer_data["email"],
                nic=officer_data.get("nic"),
                role="government",
                hospital_id=None,
                hospital_name="Department of Health",
                created_at=officer_data.get("created_at"),
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="User not found in system.",
    )
