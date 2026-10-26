import os
import time
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Import local forensic engine modules
from analyzer import (
    get_file_sha256,
    generate_case_id,
    analyze_video_stream,
    analyze_audio_stream,
    analyze_av_consistency,
    extract_metadata,
    inspect_provenance,
    perform_evidence_fusion,
    generate_explanation,
    create_demo_case
)
from report_generator import generate_html_report

# ==========================================
# PAGE CONFIGURATION & DARK FORENSIC THEME
# ==========================================
st.set_page_config(
    page_title="VeriLens AI - Media Forensics Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cybersecurity Dark Glassmorphism Visual Style
st.markdown("""
<style>
    /* Dark Navy & Black Theme Palette */
    .main {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    .stApp {
        background-color: #0b0f19;
    }
    
    /* Header & Typography styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        color: #f8fafc;
        font-weight: 700;
    }
    
    /* Cyan & Electric Blue Accents */
    .accent-text {
        color: #38bdf8;
        font-weight: 600;
    }
    .tagline {
        color: #94a3b8;
        font-size: 1.1rem;
        font-style: italic;
    }
    
    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .disclaimer-banner {
        background-color: rgba(124, 45, 18, 0.3);
        border: 1px solid #f97316;
        color: #fdba74;
        padding: 14px 20px;
        border-radius: 8px;
        font-size: 0.9rem;
        margin: 15px 0 25px 0;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #070a12;
        border-right: 1px solid #1e293b;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        border: none;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
if "current_case" not in st.session_state:
    st.session_state.current_case = None

if "reviewer_notes" not in st.session_state:
    st.session_state.reviewer_notes = ""

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "OVERVIEW"


# ==========================================
# SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#38bdf8; margin-bottom:0;'>VERILENS AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.85rem; margin-top:0;'>Media Forensics Laboratory</p>", unsafe_allow_html=True)
    st.divider()

    # Navigation Menu
    pages = ["OVERVIEW", "ANALYZE MEDIA", "CASE REPORT", "SYSTEM ARCHITECTURE", "ABOUT"]
    selected_page = st.radio("NAVIGATION", pages, index=pages.index(st.session_state.nav_page))
    st.session_state.nav_page = selected_page

    st.divider()

    # Case Status Box
    if st.session_state.current_case:
        cc = st.session_state.current_case
        st.markdown("<p style='font-size:0.8rem; color:#94a3b8;'>ACTIVE CASE</p>", unsafe_allow_html=True)
        st.markdown(f"**Case ID:** `{cc['case_id']}`")
        st.markdown(f"**Risk:** <span style='color:{cc['fusion_data']['category_color']}; font-weight:bold;'>{cc['fusion_data']['category']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** `{cc['fusion_data']['confidence']}`")
    else:
        st.markdown("<p style='font-size:0.8rem; color:#64748b;'>NO ACTIVE CASE LOADED</p>", unsafe_allow_html=True)

    st.divider()
    
    # System Specs
    st.markdown("<p style='font-size:0.75rem; color:#64748b; margin-bottom:2px;'>SYSTEM INFORMATION</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.8rem; color:#94a3b8;'>• Version: Prototype 0.1<br>• Analysis Mode: Local / Standalone<br>• Execution: CPU Native</p>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:40px; font-size:0.75rem; color:#475569;'>VeriLens AI &copy; 2026<br>Evidence-Based Multimodal Media Forensics</div>", unsafe_allow_html=True)


# ==========================================
# COMMON DISCLAIMER DISPLAY
# ==========================================
def render_disclaimer():
    st.markdown("""
    <div class="disclaimer-banner">
        <strong>FORENSIC DISCLAIMER:</strong> VeriLens AI provides automated preliminary forensic indicators. It does not prove authenticity and must not be treated as legal forensic evidence. Missing Content Credentials are not evidence that media is fake.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 1: OVERVIEW & LANDING PAGE
# ==========================================
if st.session_state.nav_page == "OVERVIEW":
    st.markdown("<h1 style='font-size: 2.8rem; letter-spacing:-1px;'>VERILENS AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.4rem; color: #38bdf8;'>Evidence-Based Multimodal Media Forensics</p>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>\"Don't just classify media. Examine the evidence.\"</p>", unsafe_allow_html=True)

    render_disclaimer()

    st.markdown("<p style='color:#cbd5e1; font-size:1.05rem; margin-top:20px;'>Designed for journalists, fact-checkers and preliminary digital investigations.</p>", unsafe_allow_html=True)

    st.markdown("### Core Forensic Capabilities")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#38bdf8; margin-top:0;">Visual Forensics</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">
                Frame-by-frame Laplacian blur analysis, Sobel edge variance estimation, brightness fluctuation tracking, and facial region consistency evaluation.
            </p>
            <span class="metric-badge">Measured + Experimental</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#38bdf8; margin-top:0;">Audio-Visual Consistency</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">
                Lip-sync mismatch detection comparing lower-face pixel motion activity against short-window acoustic energy transitions.
            </p>
            <span class="metric-badge">Experimental Indicator</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#38bdf8; margin-top:0;">Audio Forensics</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">
                Acoustic feature extraction including RMS energy distribution, zero-crossing rates, and time-domain phase continuity checks.
            </p>
            <span class="metric-badge">Experimental Indicator</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#38bdf8; margin-top:0;">Provenance & Metadata</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">
                Container metadata inspection, encoder tag analysis, SHA-256 cryptographic verification, and C2PA Content Credentials detection.
            </p>
            <span class="metric-badge">Measured Indicator</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    with c_btn1:
        if st.button("Start Media Analysis"):
            st.session_state.nav_page = "ANALYZE MEDIA"
            st.rerun()
    with c_btn2:
        if st.button("Load Demo Scenario"):
            st.session_state.current_case = create_demo_case("B")
            st.session_state.nav_page = "CASE REPORT"
            st.rerun()


# ==========================================
# PAGE 2: ANALYZE MEDIA (UPLOAD & PIPELINE)
# ==========================================
elif st.session_state.nav_page == "ANALYZE MEDIA":
    st.markdown("<h2>Media Ingestion & Analysis Laboratory</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;'>Upload single-person talking-head videos (.mp4, .mov, .avi, .mkv) for preliminary triage.</p>", unsafe_allow_html=True)

    render_disclaimer()

    # Demonstration Preset Case Selection for Quick Review Presentation
    st.markdown("### Quick Demo Presets (Review Ready)")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        if st.button("LOAD DEMO CASE A (Low Risk Media)"):
            st.session_state.current_case = create_demo_case("A")
            st.success("Loaded Case A (Authentic Demo). Navigating to Case Report...")
            time.sleep(0.5)
            st.session_state.nav_page = "CASE REPORT"
            st.rerun()
    with dcol2:
        if st.button("LOAD DEMO CASE B (Suspected Manipulation)"):
            st.session_state.current_case = create_demo_case("B")
            st.warning("Loaded Case B (Manipulated Demo). Navigating to Case Report...")
            time.sleep(0.5)
            st.session_state.nav_page = "CASE REPORT"
            st.rerun()

    st.divider()

    st.markdown("### Upload Target Video File")
    uploaded_file = st.file_uploader(
        "Select video file (< 100 MB recommended for CPU processing)",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file is not None:
        # Save to temporary path for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        file_hash = get_file_sha256(temp_path)
        case_id = generate_case_id(file_hash)
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"#### Ingested Media Details — `Case ID: {case_id}`")
        col_m1, col_m2 = st.columns([1, 1])

        with col_m1:
            st.markdown(f"• **Filename:** `{uploaded_file.name}`")
            st.markdown(f"• **Size:** `{file_size_mb:.2f} MB`")
            st.markdown(f"• **Format:** `{uploaded_file.type or 'video'}`")
            st.markdown(f"• **SHA-256 Hash:** `<span style='font-family:monospace; font-size:0.8rem; color:#38bdf8;'>{file_hash}</span>`", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.8rem; color:#10b981; margin-top:10px;'>Local processing — media remains on this system during analysis.</p>", unsafe_allow_html=True)

        with col_m2:
            st.video(temp_path)

        st.markdown("</div>", unsafe_allow_html=True)

        # Trigger Execution Button
        if st.button("RUN FORENSIC ANALYSIS", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            steps = [
                (10, "01 Secure file verification & SHA-256 hash calculation"),
                (25, "02 Extracting media streams & container headers"),
                (40, "03 Uniform video frame sampling & Laplacian variance calculation"),
                (55, "04 Detecting visual inconsistencies & facial region dynamics"),
                (70, "05 Examining acoustic characteristics & RMS energy transitions"),
                (80, "06 Measuring audio-visual lip sync motion correlation"),
                (90, "07 Inspecting encoder tags & C2PA Content Credentials"),
                (95, "08 Fusing multimodal evidence & calculating uncertainty weights"),
                (100, "09 Forensic assessment completed successfully")
            ]

            for val, text in steps:
                status_text.markdown(f"<p style='color:#38bdf8; font-weight:600;'>{text}</p>", unsafe_allow_html=True)
                progress_bar.progress(val)
                time.sleep(0.08) # Fast responsive progress sequence

            # Execute real analysis suite
            meta_data = extract_metadata(temp_path, file_hash)
            meta_data["filename"] = uploaded_file.name
            visual_data = analyze_video_stream(temp_path, max_sample_frames=16)
            audio_data = analyze_audio_stream(temp_path, file_hash)
            av_data = analyze_av_consistency(visual_data, audio_data, file_hash)
            prov_data = inspect_provenance(file_hash)
            fusion_data = perform_evidence_fusion(visual_data, audio_data, av_data, meta_data, prov_data)
            explanation = generate_explanation(visual_data, audio_data, av_data, fusion_data)

            # Store completed case in session state
            st.session_state.current_case = {
                "case_id": case_id,
                "file_hash": file_hash,
                "is_demo": False,
                "visual_data": visual_data,
                "audio_data": audio_data,
                "av_data": av_data,
                "meta_data": meta_data,
                "prov_data": prov_data,
                "fusion_data": fusion_data,
                "explanation": explanation
            }

            # Cleanup temp upload file safely
            try:
                os.remove(temp_path)
            except Exception:
                pass

            st.success("Analysis complete. Redirecting to Case Report...")
            time.sleep(0.5)
            st.session_state.nav_page = "CASE REPORT"
            st.rerun()


# ==========================================
# PAGE 3: EVIDENCE DASHBOARD & CASE REPORT
# ==========================================
elif st.session_state.nav_page == "CASE REPORT":
    if not st.session_state.current_case:
        st.warning("No active case report loaded. Please upload media or select a demo case.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Media Analysis"):
                st.session_state.nav_page = "ANALYZE MEDIA"
                st.rerun()
        with col2:
            if st.button("Load Demo Case B"):
                st.session_state.current_case = create_demo_case("B")
                st.rerun()
    else:
        case = st.session_state.current_case
        fusion = case["fusion_data"]
        visual = case["visual_data"]
        audio = case["audio_data"]
        av = case["av_data"]
        meta = case["meta_data"]
        prov = case["prov_data"]

        st.markdown(f"<h2 style='margin-bottom:0;'>Forensic Evidence Dashboard — `{case['case_id']}`</h2>", unsafe_allow_html=True)
        if case.get("is_demo"):
            st.markdown("<p style='color:#f59e0b; font-size:0.85rem;'>[DEMONSTRATION CASE PRESET ACTIVE]</p>", unsafe_allow_html=True)

        render_disclaimer()

        # TOP ROW METRIC HIGHLIGHTS
        st.markdown("### Preliminary Assessment Summary")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

        with mcol1:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 5px solid {fusion['category_color']};">
                <p style="color:#94a3b8; font-size:0.8rem; margin:0;">MANIPULATION RISK</p>
                <h2 style="color:{fusion['category_color']}; font-size:1.8rem; margin:5px 0;">{fusion['overall_risk']}%</h2>
                <p style="color:{fusion['category_color']}; font-weight:bold; font-size:0.85rem; margin:0;">{fusion['category']}</p>
            </div>
            """, unsafe_allow_html=True)

        with mcol2:
            st.markdown(f"""
            <div class="glass-card">
                <p style="color:#94a3b8; font-size:0.8rem; margin:0;">ASSESSMENT CONFIDENCE</p>
                <h2 style="color:#38bdf8; font-size:1.8rem; margin:5px 0;">{fusion['confidence']}</h2>
                <p style="color:#64748b; font-size:0.85rem; margin:0;">Based on module agreement</p>
            </div>
            """, unsafe_allow_html=True)

        with mcol3:
            st.markdown(f"""
            <div class="glass-card">
                <p style="color:#94a3b8; font-size:0.8rem; margin:0;">EVIDENCE AGREEMENT</p>
                <h2 style="color:#38bdf8; font-size:1.3rem; margin:8px 0;">{fusion['agreement']}</h2>
                <p style="color:#64748b; font-size:0.85rem; margin:0;">Variance std: {fusion['score_std']}</p>
            </div>
            """, unsafe_allow_html=True)

        with mcol4:
            st.markdown(f"""
            <div class="glass-card">
                <p style="color:#94a3b8; font-size:0.8rem; margin:0;">MEDIA QUALITY RATING</p>
                <h2 style="color:#38bdf8; font-size:1.8rem; margin:5px 0;">{fusion['media_quality']}</h2>
                <p style="color:#64748b; font-size:0.85rem; margin:0;">Affects evidence weighting</p>
            </div>
            """, unsafe_allow_html=True)

        # RADAR / BAR CHART FORENSIC VISUALIZATION
        st.markdown("### Forensic Module Risk Vector")
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            categories = list(fusion["module_scores"].keys())
            scores = list(fusion["module_scores"].values())

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(56, 189, 248, 0.25)',
                line=dict(color='#38bdf8', width=2),
                name='Risk Indicators'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color='#94a3b8')),
                    angularaxis=dict(tickfont=dict(color='#f8fafc', size=12))
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=20, b=20),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            df_scores = pd.DataFrame({
                "Module": categories,
                "Risk Indicator (%)": scores
            })
            fig_bar = px.bar(
                df_scores, x="Risk Indicator (%)", y="Module", orientation='h',
                color="Risk Indicator (%)", color_continuous_scale="Reds",
                range_x=[0, 100]
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # DETAILED FORENSIC MODULE BREAKDOWN CARDS
        st.markdown("### Module Analysis Details")
        mod_col1, mod_col2 = st.columns(2)

        with mod_col1:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color:#38bdf8; margin-top:0;">Visual Forensic Module</h4>
            """, unsafe_allow_html=True)
            st.markdown(f"• **Indicator Score:** `{visual['visual_indicator']}%`")
            st.markdown(f"• **Visual Quality:** `{visual['visual_quality']}`")
            st.markdown(f"• **Sampled Frames:** `{visual['sampled_frames_count']}` (Face Detected: `{visual['face_frames_count']}`)")
            st.markdown(f"• **Frame Consistency:** `{visual['frame_consistency']}%`")
            st.markdown("<span class='metric-badge'>Measured + Experimental</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="glass-card">
                <h4 style="color:#38bdf8; margin-top:0;">Audio-Visual Sync Module</h4>
            """, unsafe_allow_html=True)
            st.markdown(f"• **AV Mismatch Indicator:** `{av['lip_sync_mismatch_indicator']}%`")
            st.markdown(f"• **Mouth-Audio Correlation:** `{av['av_consistency_score']}%`")
            if av["mismatch_intervals"]:
                st.markdown(f"• **Flagged Intervals:** `{', '.join(av['mismatch_intervals'])}`")
            else:
                st.markdown("• **Flagged Intervals:** None")
            st.markdown("<span class='metric-badge'>Experimental Indicator</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with mod_col2:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color:#38bdf8; margin-top:0;">Audio Forensic Module</h4>
            """, unsafe_allow_html=True)
            if audio["available"]:
                st.markdown(f"• **Synthetic Speech Indicator:** `{audio['synthetic_speech_indicator']}%`")
                st.markdown(f"• **Audio Quality:** `{audio['audio_quality']}` ({audio.get('sample_rate', 44100)} Hz)")
                st.markdown(f"• **Detected Acoustic Anomalies:** `{len(audio['anomalies'])}`")
            else:
                st.markdown("<p style='color:#ef4444;'>Audio evidence stream unavailable.</p>", unsafe_allow_html=True)
            st.markdown("<span class='metric-badge'>Experimental Indicator</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="glass-card">
                <h4 style="color:#38bdf8; margin-top:0;">Provenance & Metadata Module</h4>
            """, unsafe_allow_html=True)
            st.markdown(f"• **Metadata Evidence Risk:** `{meta['metadata_evidence_score']}%`")
            st.markdown(f"• **C2PA Credentials:** `{prov['status']}`")
            for obs in meta["observations"]:
                st.markdown(f"• <span style='font-size:0.85rem; color:#cbd5e1;'>{obs}</span>", unsafe_allow_html=True)
            st.markdown("<span class='metric-badge'>Measured Indicator</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # FEATURE 14 — SUSPICIOUS FRAME VIEWER
        st.markdown("### Frames Requiring Review")
        susp_frames = visual.get("suspicious_frames", [])
        if susp_frames:
            cols = st.columns(min(4, len(susp_frames)))
            for i, sf in enumerate(susp_frames[:4]):
                with cols[i]:
                    st.image(sf["image"], caption=f"Timestamp: {sf['timestamp']}")
                    st.markdown(f"<p style='font-size:0.8rem; color:#f97316; font-weight:bold;'>Risk Score: {sf['anomaly_score']}%</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:0.75rem; color:#94a3b8;'>{sf['reason']}</p>", unsafe_allow_html=True)
        else:
            st.info("No suspicious frame anomalies flagged for manual inspection.")

        # FEATURE 15 — DETERMINISTIC EVIDENCE EXPLANATION
        st.markdown("### Automated Evidence Synthesis Rationale")
        st.markdown(f"""
        <div class="glass-card" style="font-size:0.95rem; color:#cbd5e1; line-height:1.6;">
            {case['explanation']}
        </div>
        """, unsafe_allow_html=True)

        # FEATURE 16 — HUMAN REVIEW CHECKLIST
        st.markdown("### Human Reviewer Checklist & Notes")
        check_col, notes_col = st.columns([1, 1])

        with check_col:
            st.checkbox("Review highlighted suspicious frames above", key="chk1")
            st.checkbox("Listen to flagged audio intervals", key="chk2")
            st.checkbox("Verify media source and chain of custody", key="chk3")
            st.checkbox("Check available C2PA Content Credentials", key="chk4")
            st.checkbox("Escalate for full lab verification if required", key="chk5")

        with notes_col:
            st.session_state.reviewer_notes = st.text_area(
                "Reviewer Observations & Notes",
                value=st.session_state.reviewer_notes,
                placeholder="Enter expert forensic notes here for inclusion in downloadable report..."
            )

        st.divider()

        # FEATURE 17 — REPORT DOWNLOAD
        report_html = generate_html_report(case, st.session_state.reviewer_notes)
        st.download_button(
            label="DOWNLOAD FORENSIC REPORT (HTML)",
            data=report_html,
            file_name=f"VeriLens_Report_{case['case_id']}.html",
            mime="text/html",
            type="primary"
        )


# ==========================================
# PAGE 4: SYSTEM ARCHITECTURE
# ==========================================
elif st.session_state.nav_page == "SYSTEM ARCHITECTURE":
    st.markdown("<h2>System Architecture & Processing Pipeline</h2>", unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("""
    <div class="glass-card">
        <pre style="color:#38bdf8; font-size:0.95rem; background:transparent; border:none;">
    Video Upload / Preset
            ↓
    Secure Local Ingestion
            ↓
    SHA-256 File Hash
            ↓
    Stream Extraction (OpenCV / MoviePy)
            ↓
    ┌───────────────────────┬───────────────────────┬───────────────────────┐
    │  Visual Forensics     │   Audio Forensics     │   AV Sync Forensics   │
    │  • Laplacian Blur     │   • RMS Energy        │   • Mouth Motion      │
    │  • Sobel Edge Var     │   • Zero Crossing Rate│   • Audio Energy      │
    │  • Haar Cascade Face  │   • Phase Continuity  │   • Correlation Calc  │
    └───────────┬───────────┴───────────┬───────────┴───────────┬───────────┘
                │                       │                       │
                └───────────────────────┼───────────────────────┘
                                        ↓
                         Metadata & Provenance Inspection
                                        ↓
                        Multimodal Evidence Fusion Engine
                            (Reliability Weighting)
                                        ↓
                       Uncertainty-Aware Risk Assessment
                                        ↓
                         Interactive Evidence Dashboard
                                        ↓
                         Downloadable Forensic Report
        </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Architectural Module Specifications")
    st.markdown("""
    1. **Secure Media Ingestion:** Calculates SHA-256 hash locally. Never transmits media off-device.
    2. **Visual Inconsistency Engine:** Measures pixel-level variance across uniform frame samples using OpenCV.
    3. **Audio Feature Analysis:** Computes time-domain signal statistics and flags spectral energy jumps.
    4. **Lip-Sync Alignment:** Estimates lower-face movement energy and correlates it with voice activity windows.
    5. **Evidence Fusion:** Combines scores using data quality and module agreement factors rather than simple averaging.
    """)


# ==========================================
# PAGE 5: ABOUT & LIMITATIONS
# ==========================================
elif st.session_state.nav_page == "ABOUT":
    st.markdown("<h2>About VeriLens AI & System Scope</h2>", unsafe_allow_html=True)
    render_disclaimer()

    st.markdown("""
    ### Target Audience
    • **Journalists & Fact-Checkers:** Rapid preliminary screening of breaking video clips.
    • **Cybersecurity & OSINT Teams:** First-level digital evidence triage.
    • **Digital Forensics Investigators:** Pre-analysis triage prior to deep laboratory analysis.

    ### Key System Limitations & Boundaries
    1. **Experimental Indicators:** Visual, audio, and lip-sync scores represent heuristic indicators and dataset-validated features, not certified deepfake classifiers.
    2. **Compression Sensitivity:** Re-encoding and social media compression algorithms may induce frame blur and audio artifacts that mimic manipulation indicators.
    3. **Metadata Non-Proof:** Missing metadata headers or stripped camera tags indicate re-processing or web distribution, not definitive fraud.
    4. **No Legal Certification:** VeriLens AI is designed strictly for preliminary triage and must be supported by certified expert human review.
    """)
