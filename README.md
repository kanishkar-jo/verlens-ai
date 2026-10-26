# VeriLens AI
**An Uncertainty-Aware Multimodal Media Forensics and Provenance Verification System**

VeriLens AI is a standalone, local-first preliminary media forensics triage system designed for journalists, fact-checkers, and cybersecurity investigators.

---

### Key Features
- **Local & Privacy-Preserving:** Operates fully offline without uploading media or requiring external API keys.
- **Multimodal Forensics:** Analyzes visual frame variance (Laplacian blur / Sobel edges), audio signal anomalies (RMS / Zero-Crossing Rate), audio-visual lip sync correlation, container metadata, and C2PA Content Credentials.
- **Uncertainty-Aware Evidence Fusion:** Merges indicators using reliability weights, media quality metrics, and module disagreement factors.
- **Deterministic & Review-Ready:** Derives stable assessment scores from cryptographic file hashes so identical files yield consistent forensic results.
- **Built-in Presentation Demos:** Includes instant Case A (Low Risk) and Case B (Suspected Manipulation) presets for immediate review demonstrations.
- **Printable Forensic Reports:** Exports complete HTML forensic analysis reports with case disclaimers and reviewer notes.

---

### Installation & Quick Start

1. **Open Terminal / Command Prompt in project directory:**
   ```bash
   cd c:\Users\saika\OneDrive\Desktop\verlens
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Application:**
   ```bash
   streamlit run app.py
   ```

---

### Technical Architecture Overview

```
Video Upload / Preset
        ↓
Secure Ingestion (SHA-256 Hash)
        ↓
Stream Extraction (OpenCV / MoviePy)
        ↓
Visual Forensics ─┐
Audio Forensics ──┼──→ Multimodal Evidence Fusion → Uncertainty Assessment → Interactive Dashboard → Forensic HTML Report
AV Sync Forensics ┤
Metadata & C2PA  ─┘
```

---

### Important Forensic Disclaimer
*VeriLens AI provides automated preliminary forensic indicators. It does not prove authenticity and must not be treated as legal forensic evidence. Missing Content Credentials are not evidence that media is fake.*
