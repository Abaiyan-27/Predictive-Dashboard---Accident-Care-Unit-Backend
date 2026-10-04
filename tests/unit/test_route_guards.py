"""Every API route requires an authenticated role, except signing in and changing a password."""
from fastapi.routing import APIRoute

from app.api.deps import RoleChecker
from app.main import app

PUBLIC = {("POST", "/api/auth/login"), ("POST", "/api/auth/change-password")}


def _checkers(dependant):
    for dep in dependant.dependencies:
        if isinstance(dep.call, RoleChecker):
            yield dep.call
        yield from _checkers(dep)


def test_every_api_route_has_a_role_guard():
    unguarded = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith("/api/"):
            continue
        for method in route.methods:
            if (method, route.path) in PUBLIC:
                continue
            if not list(_checkers(route.dependant)):
                unguarded.append(f"{method} {route.path}")
    assert unguarded == []


def test_mutating_routes_are_staff_only():
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/api/") and route.methods & {"POST", "PATCH", "PUT", "DELETE"}:
            if (next(iter(route.methods)), route.path) in PUBLIC:
                continue
            roles = {role for checker in _checkers(route.dependant) for role in checker.allowed_roles}
            assert roles and "government" not in roles, route.path


def test_health_routes_are_the_only_unauthenticated_reads():
    # Only routes the application defines (some tests register temporary probe routes).
    public_reads = sorted(
        r.path for r in app.routes
        if isinstance(r, APIRoute) and not r.path.startswith("/api/") and r.endpoint.__module__ == "app.main"
    )
    assert public_reads == ["/health", "/health/database-schema", "/health/models"]
