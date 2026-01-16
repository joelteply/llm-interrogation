"""
Integration test: App startup.

This test verifies the app can import and start without crashing.
This is the most basic integration test - if it fails, nothing works.
"""

import pytest


class TestAppStartup:
    """Verify app can start."""

    def test_app_imports(self):
        """App module imports without error."""
        # This import triggers:
        # - All blueprint registrations
        # - Migration function call
        # - Worker startup
        import app
        assert app.app is not None

    def test_flask_app_configured(self):
        """Flask app has required configuration."""
        import app

        # Check blueprints registered
        blueprint_names = list(app.app.blueprints.keys())
        assert "projects" in blueprint_names
        assert "probe" in blueprint_names
        assert "analyze" in blueprint_names
        assert "intel" in blueprint_names

    def test_routes_exist(self):
        """Core routes are registered."""
        import app

        # Get all registered routes
        rules = [rule.rule for rule in app.app.url_map.iter_rules()]

        # Check essential routes exist
        assert "/" in rules
        assert "/api/projects" in rules

    def test_app_test_client(self):
        """Can create test client and hit index."""
        import app

        client = app.app.test_client()
        response = client.get("/")

        # Should return 200 (either HTML template or static file)
        assert response.status_code == 200

    def test_api_projects_endpoint(self):
        """Projects API responds."""
        import app

        client = app.app.test_client()
        response = client.get("/api/projects")

        assert response.status_code == 200
        assert response.content_type == "application/json"


class TestMigration:
    """Verify migration function works."""

    def test_migrate_function_exists(self):
        """Migration function is importable."""
        from routes.project_storage import migrate_all_old_projects
        assert callable(migrate_all_old_projects)

    def test_migrate_returns_int(self):
        """Migration returns count."""
        from routes.project_storage import migrate_all_old_projects
        result = migrate_all_old_projects()
        assert isinstance(result, int)
        assert result >= 0
