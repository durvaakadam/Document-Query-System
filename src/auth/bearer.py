from fastapi import HTTPException, Header, status
from typing import Optional

# Replace with actual token given in HackRx dashboard
REQUIRED_BEARER_TOKEN = "f5f88484be980678f9c1a168de07125a410ae003e99158d783e36a64810434c4"

async def verify_bearer_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != REQUIRED_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token",
        )
