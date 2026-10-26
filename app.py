"""Streamlit dashboard: live recognition, consented registration, user management.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
from dataclasses import replace

import streamlit as st

st.set_page_config(page_title="Face Recognition System", page_icon="🙂", layout="wide")

try:
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401

    from config import Settings, privacy_notice_text
    from database import Database
    from errors import AppError, CameraError, ConfigError, ModelFileError
    from face_detector import FaceDetector
    from face_recognizer import FaceRecognizer
    from pipeline import RecognitionPipeline
    from registration import Registrar
    from utils import decode_image, open_camera, setup_logging, verify_password
except ImportError as exc:  # missing Python dependency
    st.error(
        f"A required Python package is missing: **{exc.name}**.\n\n"
        "Activate your virtual environment and run `pip install -r requirements.txt`."
    )
    st.stop()

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return Settings.from_env()


@st.cache_resource(show_spinner="Loading models...")
def load_services(_settings: Settings):
    db = Database(_settings.db_path, _settings.embeddings_dir)
    detector = FaceDetector(
        _settings.detector_model_path,
        _settings.detect_score_threshold,
        _settings.detect_nms_threshold,
        _settings.detect_top_k,
        _settings.detect_input_width,
    )
    recognizer = FaceRecognizer(_settings.recognizer_model_path)
    return db, detector, recognizer


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "running": False,
        "is_admin": False,
        "failed_logins": 0,
        "locked_until": 0.0,
        "samples": [],  # embeddings only - never raw images
        "cam_key": 0,
        "upload_key": 0,
        "flash_reg": None,
        "flash_admin": None,
        "reg_name": "",
        "reg_consent": False,
        "purged": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def show_flash(key: str) -> None:
    flash = st.session_state.get(key)
    if flash:
        level, message = flash
        getattr(st, level)(message)
        st.session_state[key] = None


def set_running(value: bool) -> None:
    st.session_state.running = value


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(settings: Settings, pipeline: RecognitionPipeline) -> None:
    ss = st.session_state
    st.sidebar.header("Administrator")
    if not settings.admin_password_hash:
        ss.is_admin = False
        st.sidebar.warning(
            "Admin access is disabled because ADMIN_PASSWORD_HASH is not set. "
            "For local use, add the line printed by `python utils.py` to the project's `.env` "
            "file and restart Streamlit. For a deployed app, add the hash to the app's "
            "Streamlit Secrets as `ADMIN_PASSWORD_HASH = \"...\"`, then reboot the app."
        )
    elif ss.is_admin:
        st.sidebar.success("Admin unlocked")
        st.sidebar.button("Lock", on_click=lambda: ss.update(is_admin=False))
    else:
        with st.sidebar.form("admin_login"):
            password = st.text_input("Admin password", type="password")
            submitted = st.form_submit_button("Unlock")
        if submitted:
            now = time.time()
            if now < ss.locked_until:
                st.sidebar.error(f"Too many attempts. Try again in {int(ss.locked_until - now)} s.")
            elif verify_password(password, settings.admin_password_hash):
                ss.is_admin = True
                ss.failed_logins = 0
                st.rerun()
            else:
                ss.failed_logins += 1
                if ss.failed_logins >= MAX_LOGIN_ATTEMPTS:
                    ss.locked_until = now + LOCKOUT_SECONDS
                    ss.failed_logins = 0
                st.sidebar.error("Incorrect password.")

    st.sidebar.header("Recognition settings")
    threshold = st.sidebar.slider(
        "Similarity threshold", 0.10, 0.90, float(settings.recognition_threshold), 0.01,
        help="Higher = stricter (fewer false accepts, more false rejects). "
             "Scores are cosine similarities, not probabilities.",
    )
    every_n = st.sidebar.slider("Recognize every N frames", 1, 15, settings.recognize_every_n_frames)
    landmarks = st.sidebar.checkbox("Show face landmarks", value=False)
    pipeline.update_runtime(threshold, every_n, landmarks)


# ---------------------------------------------------------------------------
# Live recognition tab
# ---------------------------------------------------------------------------
def render_live(settings: Settings, pipeline: RecognitionPipeline) -> None:
    ss = st.session_state
    st.caption(
        "Scores shown next to names are cosine similarities. They are **not** calibrated "
        "probabilities and do not guarantee identity."
    )
    col_start, col_stop, _ = st.columns([1, 1, 4])
    col_start.button("▶ Start camera", on_click=set_running, args=(True,), disabled=ss.running)
    col_stop.button("■ Stop camera", on_click=set_running, args=(False,), disabled=not ss.running)

    m_faces, m_fps, m_users = st.columns(3)
    faces_ph, fps_ph = m_faces.empty(), m_fps.empty()
    m_users.metric("Enrolled users", len(pipeline.gallery))
    status_ph = st.empty()
    frame_ph = st.empty()
    table_ph = st.empty()

    faces_ph.metric("Faces detected", 0)
    fps_ph.metric("FPS", "0.0")

    if not ss.running:
        status_ph.info("Camera is stopped. Press **Start camera** to begin.")
        return

    try:
        cap = open_camera(settings.camera_index, settings.camera_width, settings.camera_height)
    except CameraError as exc:
        ss.running = False
        status_ph.error(str(exc))
        return

    pipeline.reset()
    failures = 0
    try:
        while ss.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 30:
                    ss.running = False
                    status_ph.error("The camera stopped delivering frames (it may have been unplugged).")
                    break
                time.sleep(0.05)
                continue
            failures = 0
            if settings.mirror_preview:
                frame = cv2.flip(frame, 1)
            result = pipeline.process(frame)
            frame_ph.image(result.frame, channels="BGR", output_format="JPEG")
            faces_ph.metric("Faces detected", result.face_count)
            fps_ph.metric("FPS", f"{result.fps:.1f}")
            status_ph.info(f"Status: {result.status}")
            table_ph.dataframe(
                [
                    {
                        "Face": f.track_id,
                        "Result": f.name if f.kind == "match" else (
                            "Unknown Person" if f.kind == "unknown" else "Analyzing"),
                        "Similarity": round(f.score, 3) if f.kind in {"match", "unknown"} else None,
                    }
                    for f in result.faces
                ],
                hide_index=True,
            )
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Registration tab
# ---------------------------------------------------------------------------
def _enroll_callback(registrar: Registrar) -> None:
    ss = st.session_state
    try:
        user = registrar.enroll(ss.reg_name, ss.reg_consent, list(ss.samples))
    except AppError as exc:
        ss.flash_reg = ("error", str(exc))
        return
    ss.flash_reg = ("success", f"Enrolled '{user.display_name}' with {user.sample_count} samples.")
    ss.samples = []
    ss.reg_name = ""
    ss.reg_consent = False


def _clear_samples() -> None:
    st.session_state.samples = []


def _handle_image_bytes(settings: Settings, registrar: Registrar, data: bytes) -> str:
    """Evaluate one image; keep only the embedding. Returns a message."""
    ss = st.session_state
    try:
        frame = decode_image(data, settings.max_image_side)
        report = registrar.evaluate_sample(frame)
    except AppError as exc:
        return f"error|{exc}"
    except Exception:  # defensive: OpenCV failures must not crash the UI
        return "error|The image could not be processed."
    if report.ok:
        ss.samples.append(report.embedding)
        return f"success|Sample {len(ss.samples)} accepted."
    return f"warning|{report.message}"


def render_registration(settings: Settings, registrar: Registrar, is_admin: bool) -> None:
    ss = st.session_state
    if settings.require_admin_for_enrollment and not is_admin:
        st.info("Enrollment requires an administrator. Unlock admin access in the sidebar.")
        return
    show_flash("flash_reg")
    if ss.running:
        st.warning(
            "The live camera is running. Press **Stop camera** before using the camera here, "
            "otherwise the webcam may be busy."
        )
    st.markdown(privacy_notice_text(settings))
    st.text_input("Display name (English letters, digits, spaces)", key="reg_name", max_chars=50)
    st.checkbox(
        "I have read the notice above and explicitly consent to the collection and storage "
        "of my face embeddings.",
        key="reg_consent",
    )

    needed, limit = settings.min_enroll_samples, settings.max_enroll_samples
    count = len(ss.samples)
    st.progress(min(count / needed, 1.0), text=f"Valid samples: {count} (need {needed}, max {limit})")

    ready = bool(ss.reg_consent and ss.reg_name.strip())
    if not ready:
        st.caption("Enter a name and give consent to enable capturing.")
    elif count >= limit:
        st.caption("Maximum number of samples reached.")
    else:
        tab_cam, tab_up = st.tabs(["Capture with webcam", "Upload photos"])
        message = None
        with tab_cam:
            st.caption("Look straight at the camera. Vary your head angle slightly between shots.")
            shot = st.camera_input("Take a photo", key=f"cam_{ss.cam_key}")
            if shot is not None:
                message = _handle_image_bytes(settings, registrar, shot.getvalue())
                ss.cam_key += 1
        with tab_up:
            files = st.file_uploader(
                "Photos (one person per photo)",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True,
                key=f"up_{ss.upload_key}",
            )
            if files:
                results = []
                for item in files:
                    if len(ss.samples) >= limit:
                        break
                    results.append(_handle_image_bytes(settings, registrar, item.getvalue()))
                message = results[-1] if len(results) == 1 else (
                    "info|Processed " + str(len(results)) + " photos. Samples now: "
                    + str(len(ss.samples)) + ". Rejected photos had no face, several faces or low quality."
                )
                ss.upload_key += 1
        if message:
            level, text = message.split("|", 1)
            ss.flash_reg = (level, text)
            st.rerun()

    col_a, col_b, _ = st.columns([1, 1, 3])
    col_a.button(
        "Enroll", type="primary", on_click=_enroll_callback, args=(registrar,),
        disabled=not (ready and count >= needed),
    )
    col_b.button("Clear samples", on_click=_clear_samples, disabled=count == 0)


# ---------------------------------------------------------------------------
# Management tab
# ---------------------------------------------------------------------------
def _delete_callback(db: Database, user_id: str) -> None:
    try:
        deleted = db.delete_user(user_id)
        st.session_state.flash_admin = (
            ("success", "Enrollment deleted permanently.") if deleted
            else ("warning", "That user no longer exists.")
        )
    except AppError as exc:
        st.session_state.flash_admin = ("error", str(exc))


def _purge_callback(db: Database) -> None:
    try:
        count = db.purge_expired()
        st.session_state.flash_admin = ("success", f"Removed {count} expired enrollment(s).")
    except AppError as exc:
        st.session_state.flash_admin = ("error", str(exc))


def render_management(settings: Settings, db: Database, is_admin: bool) -> None:
    show_flash("flash_admin")
    if not is_admin:
        st.info(
            f"{db.count_users()} user(s) enrolled. Unlock admin access in the sidebar to view "
            "or delete enrollments."
        )
        return
    users = db.list_users()
    if not users:
        st.write("No users are enrolled yet.")
        return
    st.dataframe(
        [
            {
                "Name": u.display_name,
                "Enrolled (UTC)": u.created_at,
                "Consent given (UTC)": u.consent_at,
                "Expires (UTC)": u.expires_at or "never",
                "Samples": u.sample_count,
            }
            for u in users
        ],
        hide_index=True,
    )
    names = {u.user_id: u.display_name for u in users}
    chosen = st.selectbox("User to delete", list(names), format_func=names.get)
    confirm = st.checkbox("I understand this permanently deletes the user's embeddings.")
    st.button(
        "Delete user permanently", on_click=_delete_callback, args=(db, chosen),
        disabled=not confirm,
    )
    st.divider()
    retention = (f"{settings.retention_days} days" if settings.retention_days
                 else "disabled (keep until deleted)")
    st.caption(f"Retention policy: {retention}.")
    st.button("Purge expired enrollments now", on_click=_purge_callback, args=(db,))


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
def render_about(settings: Settings) -> None:
    st.markdown(privacy_notice_text(settings))
    st.markdown(
        """
**How it works**

1. **Detection** - YuNet finds face boxes and 5 landmarks (eyes, nose, mouth corners).
2. **Alignment + embedding** - SFace warps each face using its landmarks and converts it
   into a 128-number vector (an *embedding*).
3. **Matching** - the vector is compared with enrolled vectors using cosine similarity.
   A face is named only if the best score passes the threshold and is not ambiguous.

**Limitations**

- Scores are **not** probabilities. Accuracy depends on lighting, pose, camera quality,
  occlusion (masks, glasses) and the people being recognized.
- This project has **not** been evaluated for accuracy or demographic fairness. Before
  relying on it, measure false acceptance and false rejection rates on your own
  consented data and tune the threshold.
- Do not use it for covert monitoring, or as the only factor for security decisions.
"""
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Real-Time Face Detection & Recognition")
    try:
        settings = get_settings()
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()
    setup_logging(settings.log_level)

    try:
        db, detector, recognizer = load_services(settings)
    except ModelFileError as exc:
        st.error(f"{exc}")
        st.code("python download_models.py", language="bash")
        st.stop()
    except AppError as exc:
        st.error(str(exc))
        st.stop()

    init_state()
    ss = st.session_state
    if not ss.purged:
        try:
            removed = db.purge_expired()
            if removed:
                st.toast(f"Removed {removed} expired enrollment(s).")
        except AppError as exc:
            st.warning(str(exc))
        ss.purged = True

    try:
        gallery = db.load_gallery()
    except AppError as exc:
        st.error(str(exc))
        st.stop()

    if "pipeline" not in ss:
        ss.pipeline = RecognitionPipeline(settings, detector, recognizer, gallery)
    pipeline: RecognitionPipeline = ss.pipeline
    pipeline.set_gallery(gallery)
    registrar = Registrar(settings, detector, recognizer, db)

    render_sidebar(settings, pipeline)

    tab_live, tab_register, tab_manage, tab_about = st.tabs(
        ["Live recognition", "Register", "Manage users", "Privacy & About"]
    )
    # The live loop blocks the script, so every other tab is rendered first.
    with tab_register:
        render_registration(settings, registrar, ss.is_admin)
    with tab_manage:
        render_management(settings, db, ss.is_admin)
    with tab_about:
        render_about(settings)
    with tab_live:
        render_live(settings, pipeline)


main()
