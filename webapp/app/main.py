"""NiceGUI entry point."""
from __future__ import annotations

from nicegui import app, ui

from .config import HOST, PORT, UPLOADS_DIR, ensure_dirs
from .job_manager import JobManager
from .tabs import megadetector as md_tab
from .tabs import speciesnet as snet_tab
from .tabs import upload as upload_tab


def build_ui(jm: JobManager) -> None:
    @ui.page("/")
    def index() -> None:
        ui.label("USDE Wildlife Webapp").classes("text-2xl font-bold m-4")
        with ui.tabs().classes("w-full") as tabs:
            t_up = ui.tab("Upload")
            t_md = ui.tab("MegaDetector")
            t_sn = ui.tab("MD + SpeciesNet")
        with ui.tab_panels(tabs, value=t_up).classes("w-full"):
            with ui.tab_panel(t_up):
                upload_tab.render()
            with ui.tab_panel(t_md):
                md_tab.render(jm)
            with ui.tab_panel(t_sn):
                snet_tab.render(jm)


def main() -> None:
    ensure_dirs()
    # Serve uploaded images so the upload-tab thumbnail grid can display them.
    app.add_media_files("/uploads", UPLOADS_DIR)
    jm = JobManager()
    jm.start()
    app.on_shutdown(jm.shutdown)
    build_ui(jm)
    ui.run(host=HOST, port=PORT, title="USDE Wildlife Webapp", reload=False, show=False)


if __name__ == "__main__":
    main()
