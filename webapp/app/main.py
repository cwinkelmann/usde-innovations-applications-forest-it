"""NiceGUI entry point."""
from __future__ import annotations

from nicegui import app, ui

from .config import HOST, PORT, THUMBS_DIR, UPLOADS_DIR, ensure_dirs
from .job_manager import JobManager
from .tabs import admin as admin_tab
from .tabs import deepfaune as df_tab
from .tabs import evaluation as eval_tab
from .tabs import map_view as map_tab
from .tabs import megadetector as md_tab
from .tabs import series as series_tab
from .tabs import speciesnet as snet_tab
from .tabs import speciesnet_full as snet_full_tab
from .tabs import upload as upload_tab


def build_ui(jm: JobManager) -> None:
    @ui.page("/")
    def index() -> None:
        ui.label("USDE Wildlife Webapp").classes("text-2xl font-bold m-4")
        with ui.tabs().classes("w-full") as tabs:
            t_up = ui.tab("Upload")
            t_md = ui.tab("MegaDetector")
            t_sn = ui.tab("MD + SpeciesNet")
            t_snf = ui.tab("SpeciesNet (full)")
            t_df = ui.tab("MD + DeepFaune")
            t_se = ui.tab("Series")
            t_mp = ui.tab("Map")
            t_ev = ui.tab("Evaluation")
            t_ad = ui.tab("Admin")
        with ui.tab_panels(tabs, value=t_up).classes("w-full"):
            with ui.tab_panel(t_up):
                upload_tab.render()
            with ui.tab_panel(t_md):
                md_tab.render(jm)
            with ui.tab_panel(t_sn):
                snet_tab.render(jm)
            with ui.tab_panel(t_snf):
                snet_full_tab.render(jm)
            with ui.tab_panel(t_df):
                df_tab.render(jm)
            with ui.tab_panel(t_se):
                series_tab.render()
            with ui.tab_panel(t_mp):
                map_tab.render()
            with ui.tab_panel(t_ev):
                eval_tab.render()
            with ui.tab_panel(t_ad):
                admin_tab.render(jm)


def _warm_lissl_index() -> None:
    """Pre-load the Lissl ground-truth CSV index in a background thread so
    the first upload doesn't pay the ~30 s cold-start penalty.
    """
    import threading

    def _load() -> None:
        try:
            from .lissl_groundtruth import _load_index

            _load_index()
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_load, daemon=True).start()


def main() -> None:
    ensure_dirs()
    _warm_lissl_index()
    # tus-js-client drives the resumable upload section in the Upload
    # tab. Loaded once, page-wide; window.tus becomes available before
    # any user interaction. shared=True so this lands on every page (the
    # whole webapp is one ui.page("/") so this is academic, but more
    # robust against future multi-page additions).
    ui.add_body_html(
        '<script src="https://cdn.jsdelivr.net/npm/tus-js-client@4.3.1/'
        'dist/tus.min.js" crossorigin="anonymous"></script>',
        shared=True,
    )
    # Serve uploaded images (full-res) and their derived JPEG thumbnails. The
    # UI prefers /thumbs/ for grids and reserves /uploads/ for the full-res
    # modal view.
    app.add_media_files("/uploads", UPLOADS_DIR)
    app.add_media_files("/thumbs", THUMBS_DIR)
    jm = JobManager()
    jm.start()
    app.on_shutdown(jm.shutdown)
    build_ui(jm)
    # reconnect_timeout: how long the client waits for a server response
    # before declaring the WebSocket dead and reloading the page. Default
    # is 3 s — too short during multi-GB ZIP uploads where the server is
    # busy reading the multipart body and Socket.IO ping/pong gets
    # starved (esp. in Firefox). 300 s lets a slow upload survive.
    ui.run(
        host=HOST,
        port=PORT,
        title="USDE Wildlife Webapp",
        reload=False,
        show=False,
        reconnect_timeout=300.0,
    )


if __name__ == "__main__":
    main()
