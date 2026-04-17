import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import tempfile
import time
import os

from detector import PPEDetector
from database import init_db, log_frame, log_violation, get_violations, get_compliance_history, get_stats

# ══════════════════════════════════════════
#  Page Config
# ══════════════════════════════════════════
st.set_page_config(
    page_title="PPE Compliance Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════
#  Custom CSS
# ══════════════════════════════════════════
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 16px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #161822;
        border-right: 1px solid #2d3250;
    }

    /* Violation badge */
    .violation-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 2px;
    }
    .badge-critical { background:#7f1d1d; color:#fca5a5; }
    .badge-high     { background:#7c2d12; color:#fdba74; }
    .badge-medium   { background:#713f12; color:#fde68a; }
    .badge-safe     { background:#14532d; color:#86efac; }

    /* Live indicator */
    .live-dot {
        display:inline-block;
        width:10px; height:10px;
        background:#22c55e;
        border-radius:50%;
        animation: pulse 1s infinite;
        margin-right:6px;
    }
    @keyframes pulse {
        0%,100% { opacity:1; }
        50%      { opacity:0.3; }
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #3b82f6;
    }

    /* Compliance score big */
    .score-big {
        font-size: 52px;
        font-weight: 800;
        text-align: center;
    }
    .score-green  { color: #22c55e; }
    .score-yellow { color: #f59e0b; }
    .score-red    { color: #ef4444; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  Init
# ══════════════════════════════════════════
init_db()

@st.cache_resource
def load_model(path):
    try:
        # Try multiple possible paths for the model
        possible_paths = [
            path,  # Original path
            os.path.join(os.getcwd(), path),  # Absolute path from current dir
            os.path.join(os.path.dirname(__file__), path),  # Relative to script dir
        ]

        model_file = None
        for p in possible_paths:
            if os.path.exists(p):
                model_file = p
                break

        if model_file is None:
            st.error(f"❌ Model file not found. Tried paths: {possible_paths}")
            st.error("Please ensure the model file is uploaded to the repository.")
            return None

        st.info(f"🔄 Loading model from: {model_file}")
        detector = PPEDetector(model_file, conf=0.45, iou=0.5)
        st.success("✅ Model loaded successfully!")
        return detector
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Please check the model file and try again.")
        return None


# ══════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🦺 PPE Monitor")
    st.markdown("---")

    # Model path
    model_path = st.text_input("Model Path", value="models/best.pt")

# ══════════════════════════════════════════
#  Mode Selection with Cloud Compatibility
# ══════════════════════════════════════════
import os
is_streamlit_cloud = os.getenv('STREAMLIT_SERVER_HEADLESS', 'false').lower() == 'true' or 'mount' in os.getcwd()

if is_streamlit_cloud:
    mode = st.radio("Mode", ["🎬 Upload Video"], index=0)
    st.info("ℹ️ **Webcam mode is not available** on Streamlit Cloud. Please use video upload instead.")
else:
    mode = st.radio("Mode", ["📹 Live Webcam", "🎬 Upload Video"], index=0)

# Settings (always available)
st.markdown("### ⚙️ Settings")
conf_thresh = st.slider("Confidence", 0.1, 0.9, 0.45, 0.05)
zone        = st.selectbox("Zone", ["Zone A", "Zone B", "Zone C", "Zone D"])

st.markdown("---")

# Stats timeframe
st.markdown("### 📊 Analytics")
hours_filter = st.selectbox("Timeframe", [1, 6, 12, 24, 48], index=3,
                            format_func=lambda x: f"Last {x}h")

st.markdown("---")
st.markdown(
    "<div style='color:#64748b;font-size:12px;text-align:center'>"
    "Smart PPE Compliance Monitor<br>Powered by YOLO26</div>",
    unsafe_allow_html=True
)


# ══════════════════════════════════════════
#  Header
# ══════════════════════════════════════════
col_title, col_live = st.columns([4, 1])
with col_title:
    st.markdown("# 🦺 Smart PPE Compliance Monitor")
with col_live:
    if mode == "📹 Live Webcam":
        st.markdown(
            "<div style='margin-top:20px'>"
            "<span class='live-dot'></span>"
            "<span style='color:#22c55e;font-weight:700'>LIVE</span>"
            "</div>",
            unsafe_allow_html=True
        )

st.markdown("---")


# ══════════════════════════════════════════
#  Load Model
# ══════════════════════════════════════════
if not Path(model_path).exists():
    st.error(f"❌ Model not found: `{model_path}` — Please check the path in the sidebar.")
    st.stop()

detector = load_model(model_path)
if detector is None:
    st.error("❌ Failed to load the model. Please check the error messages above.")
    st.stop()

detector.conf = conf_thresh


# ══════════════════════════════════════════
#  Stats Row
# ══════════════════════════════════════════
stats = get_stats(hours=hours_filter)

c1, c2, c3, c4 = st.columns(4)
with c1:
    score = stats["avg_compliance"]
    color = "score-green" if score >= 80 else ("score-yellow" if score >= 50 else "score-red")
    st.markdown(
        f"<div class='score-big {color}'>{score:.0f}%</div>"
        f"<div style='text-align:center;color:#94a3b8;font-size:13px'>Avg Compliance</div>",
        unsafe_allow_html=True
    )
with c2:
    st.metric("🚨 Total Violations", stats["total_violations"])
with c3:
    top_viol = max(stats["by_type"], key=stats["by_type"].get) if stats["by_type"] else "—"
    st.metric("⚠️ Top Violation", top_viol)
with c4:
    st.metric("🕐 Timeframe", f"Last {hours_filter}h")

st.markdown("---")


# ══════════════════════════════════════════
#  Main Layout
# ══════════════════════════════════════════
col_video, col_info = st.columns([3, 2])

with col_video:
    st.markdown("<div class='section-header'>📷 Detection Feed</div>", unsafe_allow_html=True)
    video_placeholder = st.empty()

with col_info:
    st.markdown("<div class='section-header'>📋 Live Status</div>", unsafe_allow_html=True)
    status_placeholder  = st.empty()
    violations_placeholder = st.empty()


# ══════════════════════════════════════════
#  Analytics Row
# ══════════════════════════════════════════
st.markdown("---")
st.markdown("<div class='section-header'>📈 Analytics</div>", unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)
chart_placeholder1 = chart_col1.empty()
chart_placeholder2 = chart_col2.empty()

# Violations Table
st.markdown("<div class='section-header'>📋 Recent Violations Log</div>", unsafe_allow_html=True)
table_placeholder = st.empty()


# ══════════════════════════════════════════
#  Helper: Draw Analytics
# ══════════════════════════════════════════
def draw_analytics():
    history = get_compliance_history(hours=hours_filter)
    stats_now = get_stats(hours=hours_filter)

    # Compliance over time
    if history:
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        fig1 = px.area(
            df, x="timestamp", y="compliance_score",
            title="Compliance Rate Over Time",
            color_discrete_sequence=["#3b82f6"],
            template="plotly_dark",
        )
        fig1.update_layout(
            paper_bgcolor="#1e2130",
            plot_bgcolor="#1e2130",
            yaxis=dict(range=[0, 105], title="Compliance %"),
            xaxis_title="",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig1.add_hline(y=80, line_dash="dash", line_color="#22c55e",
                       annotation_text="Target 80%")
        chart_placeholder1.plotly_chart(fig1, use_container_width=True)

    # Violations by type
    if stats_now["by_type"]:
        df2 = pd.DataFrame(
            list(stats_now["by_type"].items()),
            columns=["Violation", "Count"]
        )
        fig2 = px.bar(
            df2, x="Violation", y="Count",
            title="Violations by Type",
            color="Count",
            color_continuous_scale="reds",
            template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="#1e2130",
            plot_bgcolor="#1e2130",
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        chart_placeholder2.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════
#  Helper: Draw Violations Table
# ══════════════════════════════════════════
def draw_violations_table():
    viols = get_violations(limit=50, hours=hours_filter)
    if viols:
        df = pd.DataFrame(viols)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
        df = df[["timestamp", "class_name", "confidence", "severity", "zone"]]
        df.columns = ["Time", "Violation", "Confidence", "Severity", "Zone"]
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.0%}")

        def color_severity(val):
            colors = {
                "CRITICAL": "background-color:#7f1d1d;color:#fca5a5",
                "HIGH":     "background-color:#7c2d12;color:#fdba74",
                "MEDIUM":   "background-color:#713f12;color:#fde68a",
            }
            return colors.get(val, "")

        # 🔥 الحل الصح
        styled = df.style.map(color_severity, subset=["Severity"])

        # 🔥 مهم جدًا (كنتِ نسيتيه)
        table_placeholder.dataframe(styled, use_container_width=True, height=250)

    else:
        table_placeholder.info("✅ No violations recorded in this timeframe!")
# ══════════════════════════════════════════
#  Callback: Log to DB each frame
# ══════════════════════════════════════════
def on_frame(result):
    log_frame(
        compliance_score=result.compliance_score,
        total_detections=len(result.detections),
        total_violations=len(result.violations),
        fps=result.fps,
    )
    for v in result.violations:
        log_violation(
            class_name=v.class_name,
            confidence=v.confidence,
            severity=v.severity,
            zone=zone,
        )


# ══════════════════════════════════════════
#  Mode: Live Webcam
# ══════════════════════════════════════════
if mode == "📹 Live Webcam":
    if is_streamlit_cloud:
        st.error("❌ Webcam access is not available on Streamlit Cloud. Please use video upload mode instead.")
        st.stop()

    start_btn = st.button("▶ Start Live Detection", type="primary", use_container_width=True)
    stop_btn  = st.button("⏹ Stop", use_container_width=True)

    if start_btn:
        frame_count = 0
        try:
            for result in detector.process_webcam(cam_index=0, callback=on_frame):
                if stop_btn:
                    break

                # Show frame
                frame_rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Status panel
                score = result.compliance_score
                s_color = "#22c55e" if score >= 80 else ("#f59e0b" if score >= 50 else "#ef4444")
                status_html = f"""
                <div style='background:#1e2130;border-radius:12px;padding:16px;'>
                <div style='font-size:42px;font-weight:800;color:{s_color};text-align:center'>
                    {score:.0f}%
                </div>
                <div style='text-align:center;color:#94a3b8;margin-bottom:12px'>Compliance Score</div>
                <div style='display:flex;justify-content:space-between;margin-top:8px'>
                    <span style='color:#94a3b8'>Detections</span>
                    <span style='color:#e2e8f0;font-weight:600'>{len(result.detections)}</span>
                </div>
                <div style='display:flex;justify-content:space-between;margin-top:4px'>
                    <span style='color:#94a3b8'>Violations</span>
                    <span style='color:#ef4444;font-weight:600'>{len(result.violations)}</span>
                </div>
                <div style='display:flex;justify-content:space-between;margin-top:4px'>
                    <span style='color:#94a3b8'>FPS</span>
                    <span style='color:#e2e8f0;font-weight:600'>{result.fps:.1f}</span>
                </div>
            </div>
            """
            status_placeholder.markdown(status_html, unsafe_allow_html=True)

            # Violations list
            if result.violations:
                vhtml = "<div style='margin-top:12px'>"
                for v in result.violations:
                    badge = {"CRITICAL": "badge-critical", "HIGH": "badge-high"}.get(
                        v.severity, "badge-medium")
                    vhtml += f"<span class='violation-badge {badge}'>⚠ {v.class_name}</span>"
                vhtml += "</div>"
                violations_placeholder.markdown(vhtml, unsafe_allow_html=True)
            else:
                violations_placeholder.markdown(
                    "<div style='color:#22c55e;margin-top:12px'>✅ All PPE Compliant</div>",
                    unsafe_allow_html=True
                )

            # Refresh analytics every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                draw_analytics()
                draw_violations_table()

        except Exception as e:
            st.error(f"❌ Webcam error: {str(e)}")
            st.error("💡 **Tip**: Webcam access requires local execution. Use video upload mode instead.")


# ══════════════════════════════════════════
#  Mode: Upload Video
# ══════════════════════════════════════════
else:
    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.success(f"✅ Video uploaded: {uploaded.name}")
        process_btn = st.button("🚀 Start Processing", type="primary", use_container_width=True)

        if process_btn:
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            progress_bar = st.progress(0)
            frame_idx    = [0]

            def on_video_frame(result):
                on_frame(result)
                frame_idx[0] += 1
                progress = min(frame_idx[0] / max(total_frames, 1), 1.0)
                progress_bar.progress(progress)

                frame_rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                score = result.compliance_score
                s_color = "#22c55e" if score >= 80 else ("#f59e0b" if score >= 50 else "#ef4444")
                status_html = f"""
                <div style='background:#1e2130;border-radius:12px;padding:16px;'>
                    <div style='font-size:42px;font-weight:800;color:{s_color};text-align:center'>
                        {score:.0f}%
                    </div>
                    <div style='text-align:center;color:#94a3b8;margin-bottom:8px'>Compliance Score</div>
                    <div style='display:flex;justify-content:space-between'>
                        <span style='color:#94a3b8'>Frame</span>
                        <span style='color:#e2e8f0'>{frame_idx[0]}/{total_frames}</span>
                    </div>
                    <div style='display:flex;justify-content:space-between;margin-top:4px'>
                        <span style='color:#94a3b8'>Violations</span>
                        <span style='color:#ef4444;font-weight:600'>{len(result.violations)}</span>
                    </div>
                    <div style='display:flex;justify-content:space-between;margin-top:4px'>
                        <span style='color:#94a3b8'>FPS</span>
                        <span style='color:#e2e8f0'>{result.fps:.1f}</span>
                    </div>
                </div>
                """
                status_placeholder.markdown(status_html, unsafe_allow_html=True)

                if frame_idx[0] % 30 == 0:
                    draw_analytics()
                    draw_violations_table()

            detector.process_video(tmp_path, callback=on_video_frame)
            os.unlink(tmp_path)

            st.success("✅ Video processing complete!")
            draw_analytics()
            draw_violations_table()


# ══════════════════════════════════════════
#  Always show analytics on load
# ══════════════════════════════════════════
draw_analytics()
draw_violations_table()
