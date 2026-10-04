"""Auth router — POST /api/auth/login"""
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.core.authorization import exact_ilike
from app.core.security import create_access_token
from app.db.session import get_auth_supabase, get_supabase
from app.schemas.auth import AuthResponse, ChangePasswordPayload, LoginCredentials
from app.schemas.user import UserOut

router = APIRouter(prefix="/auth", tags=["Auth"])

PASSWORD_CHANGE_REQUIRED = "PASSWORD_CHANGE_REQUIRED"



def _must_change_password(auth_user) -> bool:
    """Accounts created by an admin carry this flag until the user sets their own password."""
    metadata = getattr(auth_user, "user_metadata", None)
    return isinstance(metadata, dict) and metadata.get("must_change_password") is True


def _sign_in(auth_supabase: Client, email: str, password: str):
    try:
        response = auth_supabase.auth.sign_in_with_password({"email": email, "password": password})
    except Exception:
        response = None
    if not response or not response.user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")
    return response


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
    auth_response = _sign_in(auth_supabase, credentials.email, credentials.password)
    if _must_change_password(auth_response.user):
        # No session until the temporary password has been replaced.
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=PASSWORD_CHANGE_REQUIRED)

    # The profile belongs to the email Supabase Auth just verified, matched
    # case-insensitively (Auth stores emails in lower case).
    email_pattern = exact_ilike(getattr(auth_response.user, "email", None) or credentials.email)
    result = supabase.table("User").select("*").ilike("email", email_pattern).execute()

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
    result = supabase.table("Government_Officer").select("*").ilike("email", email_pattern).execute()

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


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(
    payload: ChangePasswordPayload,
    auth_supabase: Client = Depends(get_auth_supabase),
    supabase: Client = Depends(get_supabase),
):
    """Replace a password (required after an admin creates the account).

    The current password is checked by Supabase Auth; the new one is set
    through the Auth admin API and the first-sign-in flag is cleared.
    """
    if payload.new_password == payload.current_password:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Choose a new password different from the current one.")
    auth_response = _sign_in(auth_supabase, payload.email, payload.current_password)
    metadata = getattr(auth_response.user, "user_metadata", None)
    supabase.auth.admin.update_user_by_id(auth_response.user.id, {
        "password": payload.new_password,
        "user_metadata": {**(metadata if isinstance(metadata, dict) else {}), "must_change_password": False},
    })
