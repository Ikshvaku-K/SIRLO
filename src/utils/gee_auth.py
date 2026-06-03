
import ee
import os
import logging

log = logging.getLogger(__name__)


def initialize_gee(project: str = None, project_id: str = None):
    """
    Initialize the Google Earth Engine API.

    Args:
        project:    GEE / Google Cloud project ID (preferred, matches ee.Initialize signature).
        project_id: Legacy alias for `project` — accepted for backwards compatibility.

    If not authenticated, this function will raise an ee.EEException.
    Run `earthengine authenticate` to create credentials.
    """
    # Resolve project — prefer explicit `project`, fall back to `project_id`,
    # then environment variables.
    resolved = (
        project
        or project_id
        or os.environ.get("DEFAULT_GEE_PROJECT")
        or os.environ.get("EARTHENGINE_PROJECT")
    )

    try:
        ee.Initialize(project=resolved)
        log.info("Google Earth Engine initialised (project=%s).", resolved or "<default>")
        print(f"Google Earth Engine initialized successfully (project={resolved or 'default'}).")
    except ee.EEException as exc:
        log.error("GEE Initialisation failed: %s", exc)
        print(f"GEE Initialization failed: {exc}")
        print("Please run `earthengine authenticate` in your terminal.")
        raise


if __name__ == "__main__":
    initialize_gee()
