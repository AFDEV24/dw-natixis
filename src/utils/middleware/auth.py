from typing import Callable

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, api_key: str):
        """
        Middleware to enforce API key authentication on incoming requests.

        Args:
            app (FastAPI): FastAPI application instance.
            api_key (str): API key used for authenticating requests.
        """
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process the incoming request and enforce API key authentication.

        Args:
            request (Request): The incoming request object.
            call_next (Callable): The next middleware or route handler.

        Returns:
            Response: The response object.

        Raises:
            HTTPException: If the API key is invalid or missing.
        """
        api_key = request.headers.get("x-api-key")
        if api_key != self.api_key:
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: Invalid API Key"},
            )
        response = await call_next(request)
        return response
