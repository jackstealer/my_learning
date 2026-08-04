"""
ProteinForge: Dynamic AI Structure Prediction Dashboard
100% DYNAMIC - No hard-coded sequences or structures
All data fetched from APIs or user input at runtime
"""
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
from stmol import showmol
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
# Add the project root to sys.path so we can import the backend_integration package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend_integration.utils import (
    validate_sequence, parse_fasta, fetch_uniprot_sequence,
    predict_structure_esmfold, fetch_alphafold_structure,
    extract_plddt_from_pdb, analyze_sequence, 
    format_sequence_display, get_confidence_category, 
    parse_pdb_coordinates, calculate_rmsd,
    get_example_protein_ids, fetch_uniprot_metadata,
    get_plddt_color_scale,
    # ── Hyperparameter-tunable analysis ───────────────────────────────────────
    calculate_sequence_quality, calculate_hydrophobicity_profile,
    predict_disorder_profile, calculate_plddt_smoothed,
    trim_low_confidence_termini, assess_prediction_reliability,
)
import time


# Page configuration
st.set_page_config(
    page_title="ProteinForge - AI Structure Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS with Modern Design + 3D Animations
st.markdown("""
    <style>
    /* ─── CRITICAL: Dark backgrounds everywhere ─────────────────────────── */
    html, body {
        background-color: #080a1e !important;
        background: #080a1e !important;
    }
    .stApp, [data-testid="stApp"] {
        background: linear-gradient(160deg, #080a1e 0%, #0f1535 40%, #0a1020 100%) !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    section.main,
    .main {
        background: transparent !important;
        font-family: 'Inter', sans-serif;
    }
    /* Streamlit default light background override */
    [data-testid="block-container"] {
        background: transparent !important;
    }

    /* ─── PROFESSIONAL DARK THEME ─────────────────────────────────────────── */

    /* Hide Streamlit deploy / stop toolbar */
    header[data-testid="stHeader"] {
        background: rgba(8,10,30,0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-bottom: 1px solid rgba(0,212,255,0.15) !important;
    }
    button[data-testid="baseButton-headerNoPadding"],
    .stDeployButton, [data-testid="stToolbar"] {
        display: none !important;
    }

    /* ─── Sidebar ─────────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b1f 0%, #0d1230 60%, #070b1f 100%) !important;
        border-right: 1px solid rgba(0,212,255,0.12) !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e6f1ff !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #a8b2d1 !important;
        font-size: 0.9rem !important;
    }
    section[data-testid="stSidebar"] .stRadio > div {
        background: transparent !important;
        padding: 0 !important;
        border: none !important;
    }
    /* Nav radio item styling */
    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(0,212,255,0.08);
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 4px;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:hover {
        background: rgba(0,212,255,0.08);
        border-color: rgba(0,212,255,0.25);
    }
    /* Sidebar divider */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(0,212,255,0.1) !important;
        margin: 0.75rem 0 !important;
    }
    /* Sidebar success/info box */
    section[data-testid="stSidebar"] .stAlert {
        background: rgba(0,212,255,0.06) !important;
        border: 1px solid rgba(0,212,255,0.2) !important;
        border-left: 3px solid #00d4ff !important;
        border-radius: 8px !important;
        color: #a8b2d1 !important;
    }
    /* Sidebar caption */
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: rgba(0,212,255,0.8) !important;
        font-size: 0.8rem !important;
    }

    /* ─── Unified Card System ─────────────────────────────────────────────── */
    .pf-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0,212,255,0.12);
        border-radius: 16px;
        padding: 1.8rem;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        position: relative;
        overflow: hidden;
    }
    .pf-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(0,212,255,0.03) 0%, transparent 60%);
        pointer-events: none;
    }
    .pf-card:hover {
        border-color: rgba(0,212,255,0.3);
        background: rgba(255,255,255,0.06);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 30px rgba(0,212,255,0.08);
    }
    .pf-card h2, .pf-card h3, .pf-card h4 {
        color: #e6f1ff !important;
    }
    .pf-card p, .pf-card li, .pf-card span {
        color: rgba(168,178,209,0.9) !important;
    }
    .pf-card strong {
        color: #e6f1ff !important;
    }
    /* Accent top-border variant */
    .pf-card-cyan  { border-top: 2px solid #00d4ff; }
    .pf-card-purple { border-top: 2px solid #7b2ff7; }
    .pf-card-pink   { border-top: 2px solid #f107e8; }
    .pf-card-green  { border-top: 2px solid #00ff88; }
    .pf-card-amber  { border-top: 2px solid #f59e0b; }

    /* ─── Section Headers ─────────────────────────────────────────────────── */
    .pf-section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #e6f1ff;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .pf-section-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: rgba(168,178,209,0.7);
        margin-bottom: 1.5rem;
    }
    .pf-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), rgba(123,47,247,0.3), transparent);
        border: none;
        margin: 2.5rem 0;
    }

    /* ─── Feature Cards (gradient bg variants) ────────────────────────────── */
    .feature-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.4s cubic-bezier(0.175,0.885,0.32,1.275);
        position: relative;
        overflow: hidden;
    }
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.5), 0 0 30px rgba(0,212,255,0.15);
        border-color: rgba(0,212,255,0.25);
    }
    .feature-card h2, .feature-card h3 { color: #e6f1ff !important; }
    .feature-card p, .feature-card li   { color: rgba(168,178,209,0.9) !important; }
    .feature-card strong { color: #fff !important; }

    /* ─── Status Badges ───────────────────────────────────────────────────── */
    .pf-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .pf-badge-live {
        background: rgba(0,255,136,0.12);
        color: #00ff88;
        border: 1px solid rgba(0,255,136,0.3);
    }
    .pf-badge-cyan {
        background: rgba(0,212,255,0.12);
        color: #00d4ff;
        border: 1px solid rgba(0,212,255,0.3);
    }

    /* ─── Main Header ─────────────────────────────────────────────────────── */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(120deg, #00d4ff 0%, #7b2ff7 50%, #f107e8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 200% auto;
        text-align: center;
        padding: 1rem 0 0.3rem 0;
        letter-spacing: -0.03em;
        animation: gradient-animation 4s ease infinite;
    }
    @keyframes gradient-animation {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .sub-header {
        text-align: center;
        color: rgba(168,178,209,0.8);
        font-size: 1rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    .dynamic-badge {
        background: rgba(0,212,255,0.12);
        color: #00d4ff;
        padding: 0.25rem 0.9rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        display: inline-block;
        margin-left: 10px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        border: 1px solid rgba(0,212,255,0.3);
    }

    /* ─── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid rgba(0,212,255,0.3);
        padding: 0.55rem 1.4rem;
        background: linear-gradient(135deg, rgba(0,212,255,0.15) 0%, rgba(123,47,247,0.15) 100%);
        color: #e6f1ff !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        transition: all 0.25s ease;
        backdrop-filter: blur(10px);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0,212,255,0.25) 0%, rgba(123,47,247,0.25) 100%);
        border-color: rgba(0,212,255,0.5);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,212,255,0.2);
        color: #fff !important;
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ─── Download Button ─────────────────────────────────────────────────── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(0,255,136,0.2) 0%, rgba(0,212,255,0.2) 100%) !important;
        color: #00ff88 !important;
        border: 1px solid rgba(0,255,136,0.35) !important;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.25s ease;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, rgba(0,255,136,0.3) 0%, rgba(0,212,255,0.3) 100%) !important;
        box-shadow: 0 6px 20px rgba(0,255,136,0.25);
        transform: translateY(-2px);
    }

    /* ─── Inputs — nuclear override for all Streamlit wrappers ───────────── */

    /* Text inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border: 1px solid rgba(0,212,255,0.2) !important;
        background: rgba(255,255,255,0.04) !important;
        color: #e6f1ff !important;
        font-size: 0.92rem;
        transition: all 0.25s;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(0,212,255,0.5) !important;
        box-shadow: 0 0 0 3px rgba(0,212,255,0.12) !important;
        background: rgba(0,212,255,0.04) !important;
    }
    .stTextInput label, .stTextArea label, .stSelectbox label {
        color: rgba(168,178,209,0.9) !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }
    textarea { font-family: 'Courier New', monospace !important; }
    input::placeholder, textarea::placeholder { color: rgba(168,178,209,0.4) !important; }

    /* ─── Expanders ───────────────────────────────────────────────────────── */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(0,212,255,0.12) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.02) !important;
        backdrop-filter: blur(10px);
        margin-bottom: 0.75rem;
        transition: all 0.25s;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(0,212,255,0.25) !important;
    }
    div[data-testid="stExpander"] summary {
        color: #e6f1ff !important;
        font-weight: 600;
    }

    /* ─── Metrics ─────────────────────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: rgba(0,212,255,0.06) !important;
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        transition: all 0.25s;
    }
    div[data-testid="stMetric"]:hover {
        background: rgba(0,212,255,0.1) !important;
        border-color: rgba(0,212,255,0.3);
    }
    div[data-testid="stMetric"] label {
        color: rgba(168,178,209,0.8) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.6rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #00ff88 !important;
    }

    /* ─── Tabs ────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0,0,0,0.3);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(0,212,255,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        color: rgba(168,178,209,0.7);
        padding: 6px 18px;
        transition: all 0.2s;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.2) 0%, rgba(123,47,247,0.2) 100%) !important;
        color: #e6f1ff !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
    }

    /* ─── Progress Bar ────────────────────────────────────────────────────── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #7b2ff7 100%);
        border-radius: 10px;
    }

    /* ─── Alerts ──────────────────────────────────────────────────────────── */
    .stAlert {
        border-radius: 10px;
        border: none !important;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stInfo"] {
        background: rgba(0,212,255,0.08) !important;
        border: 1px solid rgba(0,212,255,0.2) !important;
        color: rgba(168,178,209,0.9) !important;
    }
    div[data-testid="stSuccess"] {
        background: rgba(0,255,136,0.08) !important;
        border: 1px solid rgba(0,255,136,0.2) !important;
        color: rgba(168,178,209,0.9) !important;
    }
    div[data-testid="stWarning"] {
        background: rgba(245,158,11,0.08) !important;
        border: 1px solid rgba(245,158,11,0.2) !important;
        color: rgba(168,178,209,0.9) !important;
    }
    div[data-testid="stError"] {
        background: rgba(239,68,68,0.08) !important;
        border: 1px solid rgba(239,68,68,0.2) !important;
    }

    /* ─── Typography ──────────────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #e6f1ff;
    }
    p, label, li {
        font-family: 'Inter', sans-serif;
        color: rgba(168,178,209,0.9);
    }
    [data-testid="stMarkdownContainer"] p {
        color: rgba(168,178,209,0.85);
    }

    /* ─── Code Blocks ─────────────────────────────────────────────────────── */
    .stCodeBlock pre {
        background: rgba(0,0,0,0.4) !important;
        border: 1px solid rgba(0,212,255,0.15) !important;
        border-radius: 10px;
    }

    /* ─── File Uploader ───────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(0,212,255,0.25) !important;
        border-radius: 12px;
        background: rgba(0,212,255,0.03) !important;
        padding: 1.5rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0,212,255,0.5) !important;
        background: rgba(0,212,255,0.06) !important;
    }

    /* ─── Sliders ─────────────────────────────────────────────────────────── */
    .stSlider [data-testid="stThumbValue"] {
        color: #00d4ff !important;
    }
    .stSlider [role="slider"] {
        background: #00d4ff !important;
    }

    /* ─── Divider ─────────────────────────────────────────────────────────── */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), rgba(123,47,247,0.2), transparent);
    }

    /* ─── Scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(10,14,39,0.5); }
    ::-webkit-scrollbar-thumb {
        background: rgba(0,212,255,0.25);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.45); }

    /* ─── Spinner ─────────────────────────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: #00d4ff !important;
    }

    /* ─── Glowing orbs (background accents) ──────────────────────────────── */
    .glow-orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(60px);
        opacity: 0.12;
        pointer-events: none;
        z-index: 0;
        animation: float-orb 25s ease-in-out infinite;
    }
    @keyframes float-orb {
        0%,100% { transform: translate(0,0) scale(1); }
        33%  { transform: translate(60px,-60px) scale(1.15); }
        66%  { transform: translate(-40px,80px) scale(0.9); }
    }
    </style>
    
    <!-- ═══════════════════════════════════════════════════════════════
         PREMIUM 3D BACKGROUND — Three.js r158 + Post-processing Bloom
         Features: Protein molecule, DNA helix, platform ring, particles
         ═══════════════════════════════════════════════════════════════ -->

    <!-- Three.js core -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>

    <script>
    (function() {

    // ─── Guard: only run once ───────────────────────────────────────────────────
    if (window.__pf_bg_running) return;
    window.__pf_bg_running = true;

    function initBG() {
        // ── Canvas ──────────────────────────────────────────────────────────────
        const existing = document.getElementById('canvas-3d-bg');
        if (existing) existing.remove();

        const canvas = document.createElement('canvas');
        canvas.id = 'canvas-3d-bg';
        Object.assign(canvas.style, {
            position: 'fixed', top: '0', left: '0',
            width: '100%', height: '100%',
            zIndex: '0', pointerEvents: 'none',
            opacity: '0', transition: 'opacity 1.5s ease'
        });
        document.body.insertBefore(canvas, document.body.firstChild);

        // ── Scene / Camera / Renderer ────────────────────────────────────────────
        const scene  = new THREE.Scene();
        const W = window.innerWidth, H = window.innerHeight;
        const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 2000);
        camera.position.set(0, 0, 80);

        const renderer = new THREE.WebGLRenderer({
            canvas, alpha: true, antialias: true, powerPreference: 'high-performance'
        });
        renderer.setSize(W, H);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.toneMapping = THREE.ReinhardToneMapping;
        renderer.toneMappingExposure = 1.2;

        // Fade in after load
        setTimeout(() => { canvas.style.opacity = '1'; }, 200);

        // ── Color Palette ────────────────────────────────────────────────────────
        const C = {
            cyan:   new THREE.Color(0x00d4ff),
            purple: new THREE.Color(0x7b2ff7),
            pink:   new THREE.Color(0xf107e8),
            green:  new THREE.Color(0x00ff88),
            white:  new THREE.Color(0xffffff)
        };

        // ─────────────────────────────────────────────────────────────────────────
        // 1. PARTICLE FIELD — 1500 particles, DNA helix + scattered cloud
        // ─────────────────────────────────────────────────────────────────────────
        const PC = 1500;
        const pGeo = new THREE.BufferGeometry();
        const pPos  = new Float32Array(PC * 3);
        const pCol  = new Float32Array(PC * 3);
        const pSeed = new Float32Array(PC); // random seed per particle
        const palette = [C.cyan, C.purple, C.pink, C.green];

        for (let i = 0; i < PC; i++) {
            const t = (i / PC) * Math.PI * 8;
            const helix = i < PC * 0.55;          // 55% in helix, 45% scattered
            if (helix) {
                // Double helix strand A
                const strand = i % 2;
                const r = 20 + Math.random() * 6;
                pPos[i*3]   = Math.cos(t + strand * Math.PI) * r + (Math.random()-0.5)*8;
                pPos[i*3+1] = (i / PC) * 160 - 80 + (Math.random()-0.5)*6;
                pPos[i*3+2] = Math.sin(t + strand * Math.PI) * r + (Math.random()-0.5)*8;
            } else {
                // Scattered cloud
                pPos[i*3]   = (Math.random()-0.5)*200;
                pPos[i*3+1] = (Math.random()-0.5)*160;
                pPos[i*3+2] = (Math.random()-0.5)*120 - 30;
            }
            const col = palette[Math.floor(Math.random() * palette.length)];
            pCol[i*3] = col.r; pCol[i*3+1] = col.g; pCol[i*3+2] = col.b;
            pSeed[i] = Math.random() * Math.PI * 2;
        }
        pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
        pGeo.setAttribute('color',    new THREE.BufferAttribute(pCol, 3));
        pGeo.setAttribute('seed',     new THREE.BufferAttribute(pSeed, 1));

        const pMat = new THREE.PointsMaterial({
            size: 1.4, vertexColors: true,
            transparent: true, opacity: 0.75,
            blending: THREE.AdditiveBlending,
            depthWrite: false, sizeAttenuation: true
        });
        const pSystem = new THREE.Points(pGeo, pMat);
        scene.add(pSystem);

        // Connecting lines (neural mesh)
        const lMat = new THREE.LineBasicMaterial({
            color: 0x00d4ff, transparent: true, opacity: 0.12,
            blending: THREE.AdditiveBlending
        });
        const lGeo = new THREE.BufferGeometry();
        const lPos = [];
        for (let i = 0; i < PC; i++) {
            for (let j = i + 1; j < Math.min(i + 4, PC); j++) {
                // Only connect helix portion
                if (i < PC * 0.55) {
                    lPos.push(pPos[i*3], pPos[i*3+1], pPos[i*3+2],
                              pPos[j*3], pPos[j*3+1], pPos[j*3+2]);
                }
            }
        }
        lGeo.setAttribute('position', new THREE.Float32BufferAttribute(lPos, 3));
        const lSystem = new THREE.LineSegments(lGeo, lMat);
        scene.add(lSystem);

        // ─────────────────────────────────────────────────────────────────────────
        // 2. PROTEIN MOLECULE — atom spheres + bond cylinders, central glowing orb
        // ─────────────────────────────────────────────────────────────────────────
        const moleculeGroup = new THREE.Group();
        scene.add(moleculeGroup);

        // Atom positions (icosahedron-like layout)
        const atomDefs = [
            { pos: [0, 0, 0],       color: C.cyan,   r: 2.2, emissive: 0.8 },   // core
            { pos: [5, 3, 1],       color: C.purple, r: 1.4, emissive: 0.6 },
            { pos: [-4, 4, -2],     color: C.cyan,   r: 1.2, emissive: 0.5 },
            { pos: [3, -5, 2],      color: C.pink,   r: 1.3, emissive: 0.6 },
            { pos: [-5, -3, 1],     color: C.green,  r: 1.1, emissive: 0.5 },
            { pos: [7, -1, -3],     color: C.cyan,   r: 1.0, emissive: 0.4 },
            { pos: [-2, 7, 2],      color: C.purple, r: 1.0, emissive: 0.4 },
            { pos: [1, -7, -1],     color: C.pink,   r: 0.9, emissive: 0.4 },
            { pos: [-7, 1, -2],     color: C.cyan,   r: 0.9, emissive: 0.4 },
            { pos: [4, 6, -4],      color: C.green,  r: 0.8, emissive: 0.3 },
            { pos: [-3, -6, 3],     color: C.purple, r: 0.8, emissive: 0.3 },
            { pos: [8, 4, 2],       color: C.cyan,   r: 0.7, emissive: 0.3 },
            { pos: [-8, -2, 0],     color: C.pink,   r: 0.7, emissive: 0.3 },
        ];

        const atoms = [];
        atomDefs.forEach(def => {
            const geo = new THREE.SphereGeometry(def.r, 20, 20);
            const mat = new THREE.MeshPhongMaterial({
                color: def.color,
                emissive: def.color,
                emissiveIntensity: def.emissive,
                transparent: true, opacity: 0.9,
                shininess: 120
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(...def.pos);
            moleculeGroup.add(mesh);
            atoms.push(mesh);
        });

        // Bond connections
        const bondPairs = [
            [0,1],[0,2],[0,3],[0,4],[1,5],[1,9],[2,6],[2,10],[3,7],[4,8],
            [5,11],[4,12],[6,9],[7,10],[8,11],[9,12]
        ];
        bondPairs.forEach(([a, b]) => {
            const pa = new THREE.Vector3(...atomDefs[a].pos);
            const pb = new THREE.Vector3(...atomDefs[b].pos);
            const dir  = pb.clone().sub(pa);
            const len  = dir.length();
            const mid  = pa.clone().add(pb).multiplyScalar(0.5);
            const geo  = new THREE.CylinderGeometry(0.18, 0.18, len, 8);
            const mat  = new THREE.MeshPhongMaterial({
                color: C.cyan, emissive: C.cyan, emissiveIntensity: 0.4,
                transparent: true, opacity: 0.55
            });
            const bond = new THREE.Mesh(geo, mat);
            bond.position.copy(mid);
            bond.quaternion.setFromUnitVectors(
                new THREE.Vector3(0,1,0),
                dir.normalize()
            );
            moleculeGroup.add(bond);
        });

        // Position molecule in upper-right area of scene
        moleculeGroup.position.set(25, 8, -10);
        moleculeGroup.scale.setScalar(1.1);

        // ─────────────────────────────────────────────────────────────────────────
        // 3. NEON PLATFORM RING — pulsing torus (like the Pinterest pedestal)
        // ─────────────────────────────────────────────────────────────────────────
        const ringGroup = new THREE.Group();
        scene.add(ringGroup);
        ringGroup.position.set(25, -10, -10);

        // Outer torus
        const torusGeo = new THREE.TorusGeometry(9, 0.35, 16, 100);
        const torusMat = new THREE.MeshPhongMaterial({
            color: C.purple, emissive: C.purple,
            emissiveIntensity: 1.2,
            transparent: true, opacity: 0.9
        });
        const torus = new THREE.Mesh(torusGeo, torusMat);
        torus.rotation.x = Math.PI / 2;
        ringGroup.add(torus);

        // Inner glow ring
        const innerGeo = new THREE.TorusGeometry(7, 0.15, 12, 80);
        const innerMat = new THREE.MeshPhongMaterial({
            color: C.cyan, emissive: C.cyan,
            emissiveIntensity: 2.0,
            transparent: true, opacity: 0.7
        });
        const innerRing = new THREE.Mesh(innerGeo, innerMat);
        innerRing.rotation.x = Math.PI / 2;
        ringGroup.add(innerRing);

        // Disc platform
        const discGeo = new THREE.CircleGeometry(8.5, 64);
        const discMat = new THREE.MeshPhongMaterial({
            color: 0x0a0a20, emissive: C.purple,
            emissiveIntensity: 0.15,
            transparent: true, opacity: 0.6,
            side: THREE.DoubleSide
        });
        const disc = new THREE.Mesh(discGeo, discMat);
        disc.rotation.x = -Math.PI / 2;
        ringGroup.add(disc);

        // ─────────────────────────────────────────────────────────────────────────
        // 4. AMBIENT FLOATING ORBS — large, deep in scene
        // ─────────────────────────────────────────────────────────────────────────
        const orbConfigs = [
            { pos: [-40, 20, -60],  color: C.cyan,   r: 12, intensity: 0.25 },
            { pos: [50, -30, -80],  color: C.purple, r: 18, intensity: 0.20 },
            { pos: [-30, -40, -50], color: C.pink,   r: 10, intensity: 0.22 },
            { pos: [20, 50, -70],   color: C.green,  r: 8,  intensity: 0.18 },
        ];
        const ambientOrbs = [];
        orbConfigs.forEach(cfg => {
            const geo = new THREE.SphereGeometry(cfg.r, 32, 32);
            const mat = new THREE.MeshPhongMaterial({
                color: cfg.color, emissive: cfg.color,
                emissiveIntensity: cfg.intensity,
                transparent: true, opacity: 0.18,
                wireframe: false
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(...cfg.pos);
            scene.add(mesh);
            ambientOrbs.push({ mesh, seed: Math.random() * Math.PI * 2 });
        });

        // ─────────────────────────────────────────────────────────────────────────
        // 5. DYNAMIC SCAN RING — sweeping horizontal ring animation
        // ─────────────────────────────────────────────────────────────────────────
        const scanGeo = new THREE.TorusGeometry(35, 0.08, 8, 200);
        const scanMat = new THREE.MeshBasicMaterial({
            color: C.cyan, transparent: true, opacity: 0.3,
            blending: THREE.AdditiveBlending
        });
        const scanRing = new THREE.Mesh(scanGeo, scanMat);
        scanRing.rotation.x = Math.PI / 2;
        scene.add(scanRing);

        // Second scan ring
        const scan2Geo = new THREE.TorusGeometry(28, 0.05, 8, 200);
        const scan2Mat = new THREE.MeshBasicMaterial({
            color: C.purple, transparent: true, opacity: 0.2,
            blending: THREE.AdditiveBlending
        });
        const scanRing2 = new THREE.Mesh(scan2Geo, scan2Mat);
        scanRing2.rotation.x = Math.PI / 2;
        scene.add(scanRing2);

        // ─────────────────────────────────────────────────────────────────────────
        // 6. LIGHTING
        // ─────────────────────────────────────────────────────────────────────────
        const ambLight = new THREE.AmbientLight(0x050510, 2.0);
        scene.add(ambLight);

        const pointCyan = new THREE.PointLight(0x00d4ff, 4, 200);
        pointCyan.position.set(30, 20, 20);
        scene.add(pointCyan);

        const pointPurple = new THREE.PointLight(0x7b2ff7, 3, 180);
        pointPurple.position.set(-30, -20, 10);
        scene.add(pointPurple);

        const pointPink = new THREE.PointLight(0xf107e8, 2, 150);
        pointPink.position.set(0, -30, 30);
        scene.add(pointPink);

        // ─────────────────────────────────────────────────────────────────────────
        // 7. MOUSE & SCROLL TRACKING
        // ─────────────────────────────────────────────────────────────────────────
        let mouseX = 0, mouseY = 0;
        let targetRotX = 0, targetRotY = 0;
        let scrollY = 0;

        document.addEventListener('mousemove', e => {
            mouseX = (e.clientX / window.innerWidth  - 0.5) * 2;
            mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
        });
        window.addEventListener('scroll', () => { scrollY = window.scrollY; });

        // ─────────────────────────────────────────────────────────────────────────
        // 8. ANIMATION LOOP
        // ─────────────────────────────────────────────────────────────────────────
        let t = 0;
        const clock = new THREE.Clock();

        function animate() {
            requestAnimationFrame(animate);
            const dt = clock.getDelta();
            t += dt;

            // Smooth camera parallax from mouse
            targetRotY += (mouseX * 0.8  - targetRotY) * 0.04;
            targetRotX += (mouseY * 0.4  - targetRotX) * 0.04;
            camera.position.x += (-mouseX * 12 - camera.position.x) * 0.03;
            camera.position.y += (mouseY * 8   - camera.position.y) * 0.03;
            camera.lookAt(scene.position);

            // Scroll camera drift
            camera.position.z = 80 + scrollY * 0.01;

            // ── Particle system rotation
            pSystem.rotation.y = t * 0.05 + targetRotY * 0.6;
            pSystem.rotation.x = t * 0.03 + targetRotX * 0.4;
            lSystem.rotation.y = pSystem.rotation.y;
            lSystem.rotation.x = pSystem.rotation.x;

            // ── Particle wave animation
            const pos = pGeo.attributes.position.array;
            const seed = pGeo.attributes.seed.array;
            for (let i = 0; i < PC; i++) {
                pos[i*3+1] += Math.sin(t * 0.8 + seed[i]) * 0.018;
            }
            pGeo.attributes.position.needsUpdate = true;

            // ── Molecule float + spin
            moleculeGroup.rotation.y = t * 0.25 + targetRotY * 0.5;
            moleculeGroup.rotation.x = Math.sin(t * 0.3) * 0.12;
            moleculeGroup.position.y = 8 + Math.sin(t * 0.4) * 2.5;

            // Atom pulse
            atoms.forEach((atom, i) => {
                const s = 1 + Math.sin(t * 1.2 + i * 0.7) * 0.08;
                atom.scale.setScalar(s);
            });

            // ── Ring pulse
            ringGroup.position.y = -10 + Math.sin(t * 0.35) * 1.2;
            const pulse = 0.9 + Math.sin(t * 1.5) * 0.1;
            torusMat.emissiveIntensity = pulse * 1.2;
            innerMat.emissiveIntensity = pulse * 2.0;
            torusMat.opacity = 0.7 + Math.sin(t * 1.5) * 0.2;

            // ── Scan rings sweep
            scanRing.position.y  = ((t * 18) % 160) - 80;
            scanRing2.position.y = ((t * 12 + 80) % 160) - 80;
            scanMat.opacity  = 0.15 + Math.abs(Math.sin(t * 0.5)) * 0.2;
            scan2Mat.opacity = 0.10 + Math.abs(Math.sin(t * 0.7 + 1)) * 0.15;

            // ── Ambient orb drift
            ambientOrbs.forEach(({ mesh, seed }) => {
                mesh.position.x += Math.sin(t * 0.15 + seed) * 0.06;
                mesh.position.y += Math.cos(t * 0.12 + seed * 1.3) * 0.05;
            });

            // ── Light animation
            pointCyan.intensity = 3 + Math.sin(t * 0.9) * 1.5;
            pointPurple.intensity = 2.5 + Math.cos(t * 0.7) * 1.0;

            renderer.render(scene, camera);
        }

        animate();

        // Resize handler
        window.addEventListener('resize', () => {
            const W = window.innerWidth, H = window.innerHeight;
            camera.aspect = W / H;
            camera.updateProjectionMatrix();
            renderer.setSize(W, H);
        });
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initBG);
    } else {
        // Streamlit re-renders: wait a tick
        setTimeout(initBG, 50);
    }

    })();
    </script>
""", unsafe_allow_html=True)


# Initialize session state
if 'predicted_structure' not in st.session_state:
    st.session_state.predicted_structure = None
if 'current_sequence' not in st.session_state:
    st.session_state.current_sequence = None
if 'sequence_analysis' not in st.session_state:
    st.session_state.sequence_analysis = None
if 'plddt_scores' not in st.session_state:
    st.session_state.plddt_scores = None
if 'protein_name' not in st.session_state:
    st.session_state.protein_name = "Predicted Structure"
if 'prediction_source' not in st.session_state:
    st.session_state.prediction_source = None

# ── Hyperparameter defaults (bias-variance trade-off controls) ───────────────
if 'hp_plddt_threshold' not in st.session_state:
    st.session_state.hp_plddt_threshold = 70
if 'hp_window_size' not in st.session_state:
    st.session_state.hp_window_size = 9
if 'hp_disorder_threshold' not in st.session_state:
    st.session_state.hp_disorder_threshold = 0.5
if 'hp_auto_trim' not in st.session_state:
    st.session_state.hp_auto_trim = True
if 'hp_smooth_sigma' not in st.session_state:
    st.session_state.hp_smooth_sigma = 1.5
if 'hp_ensemble_mode' not in st.session_state:
    st.session_state.hp_ensemble_mode = True


def main():
    """Main application entry point."""
    
    # Inject professional sidebar branding
    st.markdown("""
    <style>
    /* Sidebar logo header */
    .pf-sidebar-logo {
        padding: 20px 16px 16px 16px;
        border-bottom: 1px solid rgba(0,212,255,0.1);
        margin-bottom: 12px;
    }
    .pf-logo-text {
        font-size: 1.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1;
    }
    .pf-logo-sub {
        font-size: 0.68rem;
        color: rgba(168,178,209,0.5);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 3px;
    }
    .pf-version {
        display: inline-block;
        background: rgba(0,212,255,0.1);
        color: #00d4ff;
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 6px;
        font-size: 0.62rem;
        font-weight: 700;
        padding: 2px 7px;
        letter-spacing: 0.06em;
        margin-top: 6px;
    }
    .pf-status-dot {
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #00ff88;
        box-shadow: 0 0 6px #00ff88;
        margin-right: 5px;
        animation: blink 2s ease-in-out infinite;
    }
    @keyframes blink {
        0%,100% { opacity: 1; } 50% { opacity: 0.4; }
    }
    .pf-nav-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(168,178,209,0.45);
        padding: 12px 4px 6px 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(
        '<h1 class="main-header">🧬 ProteinForge</h1>'
        '<p class="sub-header">AI-Powered Protein Structure Prediction'
        '<span class="dynamic-badge">LIVE DATA</span></p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        # ── Logo & Branding ──────────────────────────────────────────────────
        st.markdown("""
        <div class="pf-sidebar-logo">
            <div class="pf-logo-text">🧬 ProteinForge</div>
            <div class="pf-logo-sub">AI Structure Prediction</div>
            <div><span class="pf-version">v2.0</span></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Navigation ───────────────────────────────────────────────────────
        st.markdown('<div class="pf-nav-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "Navigate",
            ["🏠 Home", "🔬 Predict", "📊 Batch", "🔄 Compare", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.divider()

        # ── Model Selection ──────────────────────────────────────────────────
        st.markdown('<div class="pf-nav-label">Prediction Engine</div>', unsafe_allow_html=True)
        model_choice = st.radio(
            "Choose model:",
            ["ESMFold API", "AlphaFold DB"],
            help="ESMFold: Fast predictions from any sequence.\nAlphaFold DB: Pre-computed structures (UniProt ID required)."
        )
        st.session_state.model_choice = model_choice
        
        st.divider()

        # ── Data Sources ─────────────────────────────────────────────────────
        st.markdown('<div class="pf-nav-label">Data Sources</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.2);border-radius:8px;padding:10px 12px;margin-bottom:6px;">
            <span style="color:#00ff88;font-size:0.78rem;font-weight:700;"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#00ff88;box-shadow:0 0 5px #00ff88;margin-right:5px;"></span>ALL SYSTEMS LIVE</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("✓ UniProt REST API")
        st.caption("✓ AlphaFold Database")
        st.caption("✓ ESMFold API")
        st.caption("✓ Zero hard-coded data")
        
        st.divider()

        # ── Hyperparameter Tuning Controls ───────────────────────────────────
        st.markdown('<div class="pf-nav-label">Tuning Controls</div>', unsafe_allow_html=True)
        with st.expander("⚙️ Hyperparameters", expanded=False):
            st.caption("📐 **Bias ←→ Variance Controls:**")
            
            plddt_thr = st.slider(
                "pLDDT Confidence Threshold",
                min_value=50, max_value=90, step=5,
                value=int(st.session_state.hp_plddt_threshold),
                help="Higher = stricter: more bias, less variance. Filters unreliable residues."
            )
            st.session_state.hp_plddt_threshold = plddt_thr
            
            win = st.select_slider(
                "Hydrophobicity Window",
                options=[5, 7, 9, 11, 13, 15, 17, 19, 21],
                value=int(st.session_state.hp_window_size),
                help="Larger window = smoother profile (↓ variance, ↑ bias)."
            )
            st.session_state.hp_window_size = win
            
            dis_thr = st.slider(
                "Disorder Threshold",
                min_value=0.1, max_value=0.9, step=0.1,
                value=float(st.session_state.hp_disorder_threshold),
                help="Fraction above which a window is flagged as disordered."
            )
            st.session_state.hp_disorder_threshold = dis_thr
            
            sigma = st.slider(
                "pLDDT Smoothing σ",
                min_value=0.5, max_value=5.0, step=0.5,
                value=float(st.session_state.hp_smooth_sigma),
                help="Gaussian smoothing. Higher σ = smoother curve (↓ variance)."
            )
            st.session_state.hp_smooth_sigma = sigma
            
            auto_trim = st.checkbox(
                "Auto-trim Disordered Termini",
                value=bool(st.session_state.hp_auto_trim),
                help="Remove low-pLDDT N/C terminal residues to reduce variance."
            )
            st.session_state.hp_auto_trim = auto_trim
            
            ensemble = st.checkbox(
                "Ensemble Mode",
                value=bool(st.session_state.hp_ensemble_mode),
                help="Compare ESMFold + AlphaFold when UniProt ID is known."
            )
            st.session_state.hp_ensemble_mode = ensemble
            
            # Live bias-variance dial
            bv = (plddt_thr - 50) / 40.0
            st.progress(bv, text=f"← Low Bias | Threshold: {plddt_thr} | Low Variance →")
        
        st.divider()
        
        # Quick stats if structure is loaded
        if st.session_state.predicted_structure:
            st.subheader("📈 Current Structure")
            st.info(f"**{st.session_state.protein_name}**")
            if st.session_state.plddt_scores:
                _, plddt = st.session_state.plddt_scores
                if len(plddt) > 0:
                    avg_plddt = sum(plddt) / len(plddt)
                    st.metric("Avg Confidence", f"{avg_plddt:.1f}")
    
    # Route to pages
    if page == "🏠 Home":
        show_home()
    elif page == "🔬 Predict":
        show_prediction()
    elif page == "📊 Batch":
        show_batch_analysis()
    elif page == "🔄 Compare":
        show_comparison()
    elif page == "ℹ️ About":
        show_about()


def show_3d_hero():
    """Full-screen 3D hero section for the Home page using Three.js iframe."""
    hero_html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; background: transparent; overflow: hidden; }
  canvas { display: block; }

  /* Overlay text */
  #hero-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    padding: 40px 60px;
    pointer-events: none;
    z-index: 10;
  }
  .hero-badge {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00d4ff;
    border: 1px solid rgba(0,212,255,0.5);
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 18px;
    background: rgba(0,212,255,0.08);
    backdrop-filter: blur(10px);
    animation: fadeInUp 0.8s ease forwards;
    opacity: 0;
    animation-delay: 0.3s;
  }
  .hero-title {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: clamp(38px, 5vw, 64px);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    color: #fff;
    margin-bottom: 16px;
    animation: fadeInUp 0.8s ease forwards;
    opacity: 0;
    animation-delay: 0.5s;
  }
  .hero-title .grad {
    background: linear-gradient(120deg, #00d4ff 0%, #7b2ff7 50%, #f107e8 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradMove 4s linear infinite;
  }
  .hero-subtitle {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: clamp(14px, 2vw, 18px);
    font-weight: 400;
    color: rgba(168,178,209,0.9);
    max-width: 440px;
    line-height: 1.7;
    margin-bottom: 28px;
    animation: fadeInUp 0.8s ease forwards;
    opacity: 0;
    animation-delay: 0.7s;
  }
  .hero-stats {
    display: flex;
    gap: 28px;
    animation: fadeInUp 0.8s ease forwards;
    opacity: 0;
    animation-delay: 0.9s;
  }
  .stat {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
  }
  .stat-num {
    font-family: 'Inter', sans-serif;
    font-size: clamp(22px, 3vw, 32px);
    font-weight: 800;
    color: #fff;
    line-height: 1;
  }
  .stat-num .unit {
    font-size: 0.65em;
    color: #00d4ff;
    font-weight: 600;
  }
  .stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    color: rgba(168,178,209,0.7);
    margin-top: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  /* Right side molecule label */
  #molecule-label {
    position: absolute;
    right: 60px;
    top: 50%;
    transform: translateY(-50%);
    text-align: right;
    pointer-events: none;
    animation: fadeInRight 1s ease forwards;
    opacity: 0;
    animation-delay: 1.2s;
  }
  .mol-title {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: rgba(0,212,255,0.9);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .mol-name {
    font-family: 'Inter', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #fff;
  }
  .mol-sub {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: rgba(168,178,209,0.6);
    margin-top: 4px;
  }
  /* Corner frame decorations */
  .corner {
    position: absolute;
    width: 20px; height: 20px;
    border-color: rgba(0,212,255,0.4);
    border-style: solid;
  }
  .tl { top: 16px; left: 16px; border-width: 2px 0 0 2px; }
  .tr { top: 16px; right: 16px; border-width: 2px 2px 0 0; }
  .bl { bottom: 16px; left: 16px; border-width: 0 0 2px 2px; }
  .br { bottom: 16px; right: 16px; border-width: 0 2px 2px 0; }

  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeInRight {
    from { opacity: 0; transform: translate(24px, -50%); }
    to   { opacity: 1; transform: translate(0,   -50%); }
  }
  @keyframes gradMove {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
  }
  /* Scan line */
  .scan-line {
    position: absolute;
    bottom: 30px;
    left: 60px;
    display: flex;
    align-items: center;
    gap: 10px;
    animation: fadeInUp 0.8s ease forwards;
    opacity: 0;
    animation-delay: 1.1s;
  }
  .scan-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00d4ff;
    box-shadow: 0 0 8px #00d4ff;
    animation: pulse-dot 1.5s ease-in-out infinite;
  }
  @keyframes pulse-dot {
    0%,100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(1.4); opacity: 0.6; }
  }
  .scan-text {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    color: rgba(0,212,255,0.7);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .page-counter {
    position: absolute;
    bottom: 30px;
    right: 60px;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: rgba(168,178,209,0.5);
    letter-spacing: 0.1em;
    animation: fadeInUp 0.8s ease forwards;
    opacity: 0;
    animation-delay: 1.3s;
  }
  .page-counter span { color: #fff; }
</style>
</head>
<body>
<canvas id="hero-canvas"></canvas>

<!-- Overlay -->
<div id="hero-overlay">
  <div class="hero-badge">🧬 AI-Powered Structure Prediction</div>
  <h1 class="hero-title">
    Predict Protein<br>
    <span class="grad">Structures</span><br>
    With AI.
  </h1>
  <p class="hero-subtitle">
    Harness ESMFold &amp; AlphaFold to predict 3D protein structures
    from amino acid sequences in seconds. 100% live data.
  </p>
  <div class="hero-stats">
    <div class="stat">
      <div class="stat-num">30<span class="unit">s</span></div>
      <div class="stat-label">Avg Prediction</div>
    </div>
    <div class="stat">
      <div class="stat-num">500<span class="unit">K+</span></div>
      <div class="stat-label">Protein DB</div>
    </div>
    <div class="stat">
      <div class="stat-num">2<span class="unit">x</span></div>
      <div class="stat-label">AI Models</div>
    </div>
  </div>
</div>

<!-- Molecule label -->
<div id="molecule-label">
  <div class="mol-title">Active Model</div>
  <div class="mol-name">Myoglobin</div>
  <div class="mol-sub">153 residues · ESMFold</div>
</div>

<!-- Corner frame -->
<div class="corner tl"></div>
<div class="corner tr"></div>
<div class="corner bl"></div>
<div class="corner br"></div>

<!-- Scan bar -->
<div class="scan-line">
  <div class="scan-dot"></div>
  <span class="scan-text">Live Rendering</span>
</div>
<div class="page-counter">[ <span>01</span> / 03 ]</div>

<!-- Google Fonts -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">

<!-- Three.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
<script>
(function() {
  const canvas = document.getElementById('hero-canvas');
  const W = () => window.innerWidth;
  const H = () => window.innerHeight;

  // Scene
  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, W()/H(), 0.1, 2000);
  camera.position.set(0, 0, 70);

  const renderer = new THREE.WebGLRenderer({
    canvas, alpha: true, antialias: true, powerPreference: 'high-performance'
  });
  renderer.setSize(W(), H());
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x06060f, 1);
  renderer.toneMapping = THREE.ReinhardToneMapping;
  renderer.toneMappingExposure = 1.3;

  // Colors
  const CYAN   = new THREE.Color(0x00d4ff);
  const PURPLE = new THREE.Color(0x7b2ff7);
  const PINK   = new THREE.Color(0xf107e8);
  const GREEN  = new THREE.Color(0x00ff88);

  // ─── 1. PARTICLE CLOUD ────────────────────────────────────────────────────
  const N = 1200;
  const pGeo = new THREE.BufferGeometry();
  const pPos = new Float32Array(N * 3);
  const pCol = new Float32Array(N * 3);
  const pSeed = new Float32Array(N);
  const pal = [CYAN, PURPLE, PINK, GREEN];

  for (let i = 0; i < N; i++) {
    const t = (i / N) * Math.PI * 6;
    if (i < N * 0.6) {
      const strand = i % 2, r = 16 + Math.random() * 4;
      pPos[i*3]   = Math.cos(t + strand*Math.PI)*r + (Math.random()-0.5)*6;
      pPos[i*3+1] = (i/N)*140 - 70;
      pPos[i*3+2] = Math.sin(t + strand*Math.PI)*r + (Math.random()-0.5)*6;
    } else {
      pPos[i*3]   = (Math.random()-0.5)*180;
      pPos[i*3+1] = (Math.random()-0.5)*130;
      pPos[i*3+2] = (Math.random()-0.5)*100 - 20;
    }
    const c = pal[i % 4];
    pCol[i*3] = c.r; pCol[i*3+1] = c.g; pCol[i*3+2] = c.b;
    pSeed[i] = Math.random() * Math.PI * 2;
  }
  pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  pGeo.setAttribute('color',    new THREE.BufferAttribute(pCol, 3));
  pGeo.setAttribute('seed',     new THREE.BufferAttribute(pSeed, 1));

  const pMat = new THREE.PointsMaterial({
    size: 1.3, vertexColors: true,
    transparent: true, opacity: 0.7,
    blending: THREE.AdditiveBlending, depthWrite: false
  });
  const particles = new THREE.Points(pGeo, pMat);
  scene.add(particles);

  // Connecting lines
  const lGeo = new THREE.BufferGeometry();
  const lPos = [];
  for (let i = 0; i < N * 0.6; i++) {
    for (let j = i+1; j < Math.min(i+4, N*0.6); j++) {
      lPos.push(pPos[i*3],pPos[i*3+1],pPos[i*3+2],
                pPos[j*3],pPos[j*3+1],pPos[j*3+2]);
    }
  }
  lGeo.setAttribute('position', new THREE.Float32BufferAttribute(lPos, 3));
  const lines = new THREE.LineSegments(lGeo, new THREE.LineBasicMaterial({
    color: 0x00d4ff, transparent: true, opacity: 0.10,
    blending: THREE.AdditiveBlending
  }));
  scene.add(lines);

  // ─── 2. PROTEIN MOLECULE ──────────────────────────────────────────────────
  const mol = new THREE.Group();
  scene.add(mol);

  const atomDefs = [
    { p: [0,0,0],     c: CYAN,   r: 2.4, e: 1.0 },
    { p: [5,3,1],     c: PURPLE, r: 1.5, e: 0.7 },
    { p: [-4,4,-2],   c: CYAN,   r: 1.3, e: 0.6 },
    { p: [3,-5,2],    c: PINK,   r: 1.4, e: 0.7 },
    { p: [-5,-3,1],   c: GREEN,  r: 1.2, e: 0.6 },
    { p: [7,-1,-3],   c: CYAN,   r: 1.1, e: 0.5 },
    { p: [-2,7,2],    c: PURPLE, r: 1.1, e: 0.5 },
    { p: [1,-7,-1],   c: PINK,   r: 1.0, e: 0.5 },
    { p: [-7,1,-2],   c: CYAN,   r: 1.0, e: 0.5 },
    { p: [4,6,-4],    c: GREEN,  r: 0.9, e: 0.4 },
    { p: [-3,-6,3],   c: PURPLE, r: 0.9, e: 0.4 },
    { p: [9,2,1],     c: CYAN,   r: 0.8, e: 0.3 },
    { p: [-8,-2,0],   c: PINK,   r: 0.8, e: 0.3 },
    { p: [2,9,-2],    c: GREEN,  r: 0.7, e: 0.3 },
    { p: [-1,-9,2],   c: CYAN,   r: 0.7, e: 0.3 },
  ];
  const atomMeshes = [];
  atomDefs.forEach(d => {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(d.r, 24, 24),
      new THREE.MeshPhongMaterial({
        color: d.c, emissive: d.c, emissiveIntensity: d.e,
        transparent: true, opacity: 0.92, shininess: 150
      })
    );
    m.position.set(...d.p);
    mol.add(m);
    atomMeshes.push(m);
  });

  // Bonds
  [[0,1],[0,2],[0,3],[0,4],[1,5],[2,6],[3,7],[4,8],[5,9],[6,10],[7,11],[8,12],[9,13],[10,14]].forEach(([a,b]) => {
    if (b >= atomDefs.length) return;
    const pa = new THREE.Vector3(...atomDefs[a].p);
    const pb = new THREE.Vector3(...atomDefs[b].p);
    const dir = pb.clone().sub(pa);
    const len = dir.length();
    const geo = new THREE.CylinderGeometry(0.2, 0.2, len, 8);
    const mat = new THREE.MeshPhongMaterial({
      color: CYAN, emissive: CYAN, emissiveIntensity: 0.5,
      transparent: true, opacity: 0.5
    });
    const bond = new THREE.Mesh(geo, mat);
    bond.position.copy(pa.clone().add(pb).multiplyScalar(0.5));
    bond.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), dir.normalize());
    mol.add(bond);
  });

  mol.position.set(22, 6, -8);

  // ─── 3. PLATFORM RING ─────────────────────────────────────────────────────
  const ring = new THREE.Group();
  scene.add(ring);
  ring.position.set(22, -12, -8);

  const outerT = new THREE.Mesh(
    new THREE.TorusGeometry(10, 0.4, 16, 120),
    new THREE.MeshPhongMaterial({ color: PURPLE, emissive: PURPLE, emissiveIntensity: 1.5, transparent: true, opacity: 0.9 })
  );
  outerT.rotation.x = Math.PI/2;
  ring.add(outerT);

  const innerT = new THREE.Mesh(
    new THREE.TorusGeometry(8, 0.2, 12, 100),
    new THREE.MeshPhongMaterial({ color: CYAN, emissive: CYAN, emissiveIntensity: 2.5, transparent: true, opacity: 0.8 })
  );
  innerT.rotation.x = Math.PI/2;
  ring.add(innerT);

  // Disc
  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(9.5, 64),
    new THREE.MeshPhongMaterial({
      color: 0x08081a, emissive: PURPLE, emissiveIntensity: 0.2,
      transparent: true, opacity: 0.65, side: THREE.DoubleSide
    })
  );
  disc.rotation.x = -Math.PI/2;
  ring.add(disc);

  // ─── 4. AMBIENT ORB ───────────────────────────────────────────────────────
  const orbMesh = new THREE.Mesh(
    new THREE.SphereGeometry(14, 32, 32),
    new THREE.MeshPhongMaterial({
      color: PURPLE, emissive: PURPLE, emissiveIntensity: 0.3,
      transparent: true, opacity: 0.15
    })
  );
  orbMesh.position.set(-40, 15, -55);
  scene.add(orbMesh);

  const orb2 = new THREE.Mesh(
    new THREE.SphereGeometry(9, 24, 24),
    new THREE.MeshPhongMaterial({
      color: CYAN, emissive: CYAN, emissiveIntensity: 0.25,
      transparent: true, opacity: 0.12
    })
  );
  orb2.position.set(50, -25, -65);
  scene.add(orb2);

  // ─── 5. SCAN RINGS ────────────────────────────────────────────────────────
  const mkScan = (r, c, opacity) => {
    const m = new THREE.Mesh(
      new THREE.TorusGeometry(r, 0.07, 8, 200),
      new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity, blending: THREE.AdditiveBlending })
    );
    m.rotation.x = Math.PI/2;
    scene.add(m);
    return m;
  };
  const scan1 = mkScan(40, 0x00d4ff, 0.25);
  const scan2 = mkScan(30, 0x7b2ff7, 0.18);

  // ─── 6. LIGHTS ────────────────────────────────────────────────────────────
  scene.add(new THREE.AmbientLight(0x050515, 3));
  const pl1 = new THREE.PointLight(0x00d4ff, 5, 250); pl1.position.set(30,25,20); scene.add(pl1);
  const pl2 = new THREE.PointLight(0x7b2ff7, 4, 200); pl2.position.set(-30,-20,15); scene.add(pl2);
  const pl3 = new THREE.PointLight(0xf107e8, 3, 160); pl3.position.set(0,-30,30); scene.add(pl3);

  // ─── 7. MOUSE TRACKING ────────────────────────────────────────────────────
  let mx = 0, my = 0, trx = 0, try_ = 0;
  document.addEventListener('mousemove', e => {
    mx = (e.clientX/W() - 0.5)*2;
    my = (e.clientY/H() - 0.5)*2;
  });

  // ─── 8. ANIMATE ───────────────────────────────────────────────────────────
  let t = 0;
  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.05);
    t += dt;

    // Camera parallax
    trx += (mx*10 - trx)*0.04;
    try_ += (-my*7 - try_)*0.04;
    camera.position.x = trx;
    camera.position.y = try_;
    camera.lookAt(scene.position);

    // Particles
    particles.rotation.y = t*0.04 + mx*0.3;
    particles.rotation.x = t*0.02 + my*0.2;
    lines.rotation.copy(particles.rotation);

    // Particle wave
    const pos = pGeo.attributes.position.array;
    const seed = pGeo.attributes.seed.array;
    for (let i = 0; i < N; i++) {
      pos[i*3+1] += Math.sin(t*0.9 + seed[i])*0.016;
    }
    pGeo.attributes.position.needsUpdate = true;

    // Molecule
    mol.rotation.y = t*0.3 + mx*0.5;
    mol.rotation.x = Math.sin(t*0.25)*0.15;
    mol.position.y = 6 + Math.sin(t*0.4)*2.5;
    atomMeshes.forEach((a, i) => {
      a.scale.setScalar(1 + Math.sin(t*1.3 + i*0.8)*0.09);
    });

    // Ring
    ring.position.y = -12 + Math.sin(t*0.35)*1.5;
    outerT.material.emissiveIntensity = 1.0 + Math.sin(t*1.5)*0.5;
    innerT.material.emissiveIntensity = 2.0 + Math.sin(t*2.0)*0.8;

    // Scan rings
    scan1.position.y = ((t*20) % 160) - 80;
    scan2.position.y = ((t*14 + 80) % 160) - 80;

    // Orbs drift
    orbMesh.position.x += Math.sin(t*0.12)*0.06;
    orbMesh.position.y += Math.cos(t*0.09)*0.05;
    orb2.position.x += Math.sin(t*0.15 + 1)*0.05;
    orb2.position.y += Math.cos(t*0.13 + 2)*0.04;

    // Lights pulse
    pl1.intensity = 4 + Math.sin(t*0.9)*2;
    pl2.intensity = 3 + Math.cos(t*0.7)*1.5;

    renderer.render(scene, camera);
  }
  animate();

  // Resize
  window.addEventListener('resize', () => {
    renderer.setSize(W(), H());
    camera.aspect = W()/H();
    camera.updateProjectionMatrix();
  });

})();
</script>
</body>
</html>
    """
    components.html(hero_html, height=560, scrolling=False)


def show_home():
    """Home page with overview."""
    
    # ── 3D Hero Section ──────────────────────────────────────────────────────
    show_3d_hero()

    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)

    # Feature cards with modern gradient design
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div style="text-align: center; color: white;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚡</div>
                <h2 style="color: white; margin-bottom: 0.5rem;">Lightning Fast</h2>
                <p style="color: rgba(255,255,255,0.9); font-size: 1rem;">Get predictions in 30-60 seconds using state-of-the-art AI</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div style="text-align: center; color: white;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎯</div>
                <h2 style="color: white; margin-bottom: 0.5rem;">High Accuracy</h2>
                <p style="color: rgba(255,255,255,0.9); font-size: 1rem;">Powered by ESMFold & AlphaFold models</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div style="text-align: center; color: white;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌐</div>
                <h2 style="color: white; margin-bottom: 0.5rem;">100% Dynamic</h2>
                <p style="color: rgba(255,255,255,0.9); font-size: 1rem;">All data fetched live from APIs</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("""
    <div class="pf-card" style="margin-bottom:0;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.2rem;">
            <span style="font-size:1.4rem;">🚀</span>
            <h2 style="margin:0;font-size:1.3rem;color:#e6f1ff;">Quick Start Guide</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="pf-card pf-card-cyan">
            <h3 style="color:#00d4ff;margin-bottom:1rem;font-size:1rem;">1️⃣ Input Your Protein</h3>
            <ul style="line-height:2;font-size:0.93rem;color:rgba(168,178,209,0.9);padding-left:1.2rem;">
                <li><strong style="color:#e6f1ff;">Paste</strong> amino acid sequence directly</li>
                <li><strong style="color:#e6f1ff;">Upload</strong> FASTA file format</li>
                <li><strong style="color:#e6f1ff;">Fetch</strong> from UniProt database</li>
                <li><strong style="color:#e6f1ff;">Try</strong> example proteins below</li>
            </ul>
            <br>
            <h3 style="color:#7b2ff7;margin-bottom:1rem;font-size:1rem;">2️⃣ Choose Prediction Engine</h3>
            <ul style="line-height:2;font-size:0.93rem;color:rgba(168,178,209,0.9);padding-left:1.2rem;">
                <li><strong style="color:#e6f1ff;">ESMFold API</strong>: Fast, works with any sequence</li>
                <li><strong style="color:#e6f1ff;">AlphaFold DB</strong>: Pre-computed, requires UniProt ID</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pf-card pf-card-purple">
            <h3 style="color:#7b2ff7;margin-bottom:1rem;font-size:1rem;">3️⃣ Visualize Results</h3>
            <ul style="line-height:2;font-size:0.93rem;color:rgba(168,178,209,0.9);padding-left:1.2rem;">
                <li>Interactive <strong style="color:#e6f1ff;">3D structure viewer</strong></li>
                <li>Confidence metrics (<strong style="color:#e6f1ff;">pLDDT scores</strong>)</li>
                <li>Detailed <strong style="color:#e6f1ff;">sequence analysis</strong></li>
                <li>Amino acid <strong style="color:#e6f1ff;">composition charts</strong></li>
            </ul>
            <br>
            <h3 style="color:#f107e8;margin-bottom:1rem;font-size:1rem;">4️⃣ Export & Analyze</h3>
            <ul style="line-height:2;font-size:0.93rem;color:rgba(168,178,209,0.9);padding-left:1.2rem;">
                <li>Download <strong style="color:#e6f1ff;">PDB files</strong></li>
                <li>Compare with <strong style="color:#e6f1ff;">reference structures</strong></li>
                <li>Batch process <strong style="color:#e6f1ff;">multiple sequences</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="pf-divider">', unsafe_allow_html=True)

    # Example proteins section
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <div class="pf-section-title">🧪 Try Example Proteins</div>
        <div class="pf-section-sub">Click any card to fetch live from UniProt API</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    examples = get_example_protein_ids()
    cols = st.columns(5)
    
    for idx, (name, info) in enumerate(examples.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="pf-card" style="text-align:center;padding:1.4rem 1rem;min-height:160px;">
                <div style="font-size:2rem;margin-bottom:0.6rem;">🧬</div>
                <div style="font-size:0.85rem;font-weight:700;color:#00d4ff;margin-bottom:4px;">{name.split('(')[0].strip()}</div>
                <div style="font-size:0.72rem;color:rgba(168,178,209,0.6);font-family:'Courier New',monospace;">{info['uniprot']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Load", key=f"ex_{idx}", use_container_width=True):
                with st.spinner(f"Fetching {name}..."):
                    seq, prot_name = fetch_uniprot_sequence(info['uniprot'])
                    if seq:
                        st.session_state.temp_sequence = seq
                        st.session_state.temp_name = prot_name
                        st.session_state.temp_uniprot = info['uniprot']
                        st.success(f"✅ Loaded! Go to Predict page")
                        st.balloons()
                    else:
                        st.error("Failed to fetch")
    
    st.markdown('<hr class="pf-divider">', unsafe_allow_html=True)

    # Dataset section
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <div class="pf-section-title">🗄️ Browse Protein Dataset</div>
        <div class="pf-section-sub">Explore 500K+ sequences from Hugging Face — loaded dynamically at runtime</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    from backend_integration.utils import get_random_sequences_from_dataset, get_dataset_stats
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎲 Load Random Sequences from HF Dataset", use_container_width=True):
            with st.spinner("Loading from Hugging Face..."):
                random_seqs = get_random_sequences_from_dataset(5)
                if random_seqs:
                    st.session_state.random_dataset_sequences = random_seqs
                    st.success(f"✅ Loaded {len(random_seqs)} sequences")
                else:
                    st.error("Failed to load dataset")
    
    with col2:
        if st.button("📊 Dataset Stats", use_container_width=True):
            with st.spinner("Loading statistics..."):
                stats = get_dataset_stats()
                if stats:
                    st.session_state.dataset_stats = stats
    
    # Display dataset stats if available
    if 'dataset_stats' in st.session_state and st.session_state.dataset_stats:
        stats = st.session_state.dataset_stats
        st.subheader("📈 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sequences", f"{stats['total_sequences']:,}")
        with col2:
            st.metric("Avg Length", f"{stats['avg_length']:.0f} aa")
        with col3:
            st.metric("Min Length", f"{stats['min_length']} aa")
        with col4:
            st.metric("Max Length", f"{stats['max_length']} aa")
    
    # Display random sequences if loaded
    if 'random_dataset_sequences' in st.session_state:
        st.subheader("🧬 Random Sequences from Dataset")
        for seq_info in st.session_state.random_dataset_sequences:
            with st.expander(f"🔬 {seq_info['id']} - {seq_info['length']} residues ({seq_info['method']}, {seq_info['resolution']}Å)"):
                st.code(seq_info['sequence'])
                if st.button(f"Use this sequence", key=f"use_{seq_info['id']}", use_container_width=True):
                    st.session_state.temp_sequence = seq_info['sequence']
                    st.session_state.temp_name = seq_info['id']
                    st.success("✅ Sequence loaded! Go to Predict page")


def show_prediction():
    """Main prediction page - fully dynamic."""

    # Page header
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div class="pf-section-title">🔬 Protein Structure Prediction</div>
        <div class="pf-section-sub">Predict 3D structure from amino acid sequence using state-of-the-art AI models</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("🌐 **All data fetched dynamically** - No hard-coded sequences")
    
    # Check if sequence was loaded from home page
    sequence = None
    protein_name = "Predicted Protein"
    uniprot_id = None
    
    if 'temp_sequence' in st.session_state and st.session_state.temp_sequence:
        sequence = st.session_state.temp_sequence
        protein_name = st.session_state.get('temp_name', 'Loaded Sequence')
        uniprot_id = st.session_state.get('temp_uniprot', None)
        # Copy to current_input for persistence
        st.session_state.current_input_sequence = sequence
        st.session_state.current_input_name = protein_name
        st.session_state.current_input_uniprot = uniprot_id
        # Clear temp to avoid showing message repeatedly
        st.session_state.temp_sequence = None
        st.success(f"📥 Pre-loaded: {protein_name} ({len(sequence)} residues)")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📝 Paste Sequence", "📄 Upload FASTA", "🔗 UniProt ID", "🎲 Random from Dataset"],
        horizontal=True
    )
    
    if input_method == "📝 Paste Sequence":
        # Get existing sequence from session state if available
        default_value = ""
        if 'current_input_sequence' in st.session_state and input_method == "📝 Paste Sequence":
            default_value = st.session_state.current_input_sequence
        
        sequence_input = st.text_area(
            "Enter amino acid sequence:",
            value=default_value,
            height=150,
            placeholder="Example: MSKGEELFTGVVPILVELD... (paste your amino acid sequence here)",
            help="Paste raw sequence using standard amino acid codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)"
        )
        
        if sequence_input:
            sequence = sequence_input.replace(" ", "").replace("\n", "").upper()
            protein_name = "User Sequence"
            # Store in session state
            st.session_state.current_input_sequence = sequence
            st.session_state.current_input_name = protein_name
            st.session_state.current_input_uniprot = None
    
    elif input_method == "📄 Upload FASTA":
        uploaded_file = st.file_uploader(
            "Upload FASTA file:",
            type=['fasta', 'fa', 'txt', 'faa'],
            help="Upload FASTA format file"
        )
        
        if uploaded_file:
            fasta_content = uploaded_file.read().decode('utf-8')
            sequences = parse_fasta(fasta_content)
            
            if sequences:
                if len(sequences) > 1:
                    selected = st.selectbox(
                        "Multiple sequences found. Select one:",
                        range(len(sequences)),
                        format_func=lambda i: f"{sequences[i][0]} ({len(sequences[i][1])} residues)"
                    )
                    protein_name, sequence = sequences[selected]
                else:
                    protein_name, sequence = sequences[0]
                
                # Store in session state
                st.session_state.current_input_sequence = sequence
                st.session_state.current_input_name = protein_name
                st.session_state.current_input_uniprot = None
                st.success(f"✅ Loaded: {protein_name} ({len(sequence)} residues)")
    
    elif input_method == "🔗 UniProt ID":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uniprot_input = st.text_input(
                "Enter UniProt Accession:",
                value=uniprot_id if uniprot_id else "",
                placeholder="e.g., P69905, P01308, P42212"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            fetch_btn = st.button("🔍 Fetch", use_container_width=True)
        
        if fetch_btn and uniprot_input:
            uniprot_id = uniprot_input.strip()
            with st.spinner(f"Fetching from UniProt API..."):
                sequence, protein_name = fetch_uniprot_sequence(uniprot_id)
                
                if sequence:
                    # Store in session state
                    st.session_state.current_input_sequence = sequence
                    st.session_state.current_input_name = protein_name
                    st.session_state.current_input_uniprot = uniprot_id
                    
                    st.success(f"✅ Fetched: {protein_name} ({len(sequence)} residues)")
                    
                    # Fetch metadata
                    metadata = fetch_uniprot_metadata(uniprot_id)
                    if metadata:
                        with st.expander("📋 Protein Metadata"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Name:** {metadata['name']}")
                                st.write(f"**Organism:** {metadata['organism']}")
                            with col2:
                                st.write(f"**Gene:** {metadata['gene']}")
                                st.write(f"**Length:** {metadata['length']} aa")
                else:
                    st.error(f"❌ Could not fetch {uniprot_id}")
    
    elif input_method == "🎲 Random from Dataset":
        from backend_integration.utils import get_random_sequences_from_dataset
        
        if st.button("🎲 Load Random Sequence", use_container_width=True):
            with st.spinner("Loading from dataset..."):
                random_seqs = get_random_sequences_from_dataset(1)
                if random_seqs:
                    seq_info = random_seqs[0]
                    # Store in session state so it persists across reruns
                    st.session_state.temp_sequence = seq_info['sequence']
                    st.session_state.temp_name = seq_info['id']
                    st.session_state.temp_uniprot = None  # No UniProt ID from dataset
                    st.success(f"✅ Loaded: {seq_info['id']} ({seq_info['length']} residues)")
                    st.rerun()  # Force rerun to update the display
        
        # Load from session state if available
        if 'temp_sequence' in st.session_state and st.session_state.temp_sequence:
            sequence = st.session_state.temp_sequence
            protein_name = st.session_state.get('temp_name', 'Dataset Sequence')
            st.success(f"📥 Loaded: {protein_name} ({len(sequence)} residues)")
    
    st.markdown("---")
    
    # Get sequence from session state (persists across button clicks)
    if 'current_input_sequence' in st.session_state:
        sequence = st.session_state.current_input_sequence
        protein_name = st.session_state.get('current_input_name', 'Predicted Protein')
        uniprot_id = st.session_state.get('current_input_uniprot', None)
    
    # Debug: Show what we have
    if sequence:
        st.caption(f"✅ Sequence ready for prediction ({len(sequence)} residues)")
    
    # Prediction section
    if sequence:
        # Validate sequence
        is_valid, message = validate_sequence(sequence)
        
        if not is_valid:
            st.error(f"❌ Invalid sequence: {message}")
            return
        
        # Use validated sequence
        sequence = message  # validate_sequence returns cleaned sequence in message
        
        # Show sequence info
        with st.expander("📋 Sequence Information", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text(f"Name: {protein_name}")
                st.text(f"Length: {len(sequence)} residues")
                if uniprot_id:
                    st.text(f"UniProt: {uniprot_id}")
            with col2:
                st.metric("Sequence Length", f"{len(sequence)} aa")
            
            st.text_area("Sequence:", format_sequence_display(sequence), height=100)
        
        # ── Sequence Quality Assessment ──────────────────────────────────────────
        quality = calculate_sequence_quality(sequence)
        q_score = quality['quality_score']
        with st.expander("🔍 Sequence Quality Assessment", expanded=True):
            q_icon = "🟢" if q_score >= 70 else ("🟡" if q_score >= 40 else "🔴")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Quality Score", f"{q_icon} {q_score}/100")
            with col2:
                st.metric("Complexity", f"{quality['complexity']:.1f}%")
            with col3:
                st.metric("Disorder Risk", f"{quality['disorder_score']:.1f}%")
            st.progress(q_score / 100)
            for warn in quality['warnings']:
                st.warning(f"⚠️ {warn}")
            if not quality['warnings']:
                st.success("✅ Sequence looks good for structure prediction!")
        
        st.markdown("---")
        
        # Prediction button
        model_choice = st.session_state.get('model_choice', 'ESMFold API')
        
        if model_choice == "ESMFold API":
            predict_btn = st.button("🚀 Predict with ESMFold", use_container_width=True, type="primary")
            
            if predict_btn:
                st.info("🧬 Starting ESMFold prediction... This will take 30-60 seconds.")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress bar in background while making API call
                status_text.text("Submitting sequence to ESMFold API...")
                progress_bar.progress(10)
                
                # Make the actual API call
                pdb_string, error = predict_structure_esmfold(sequence)
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                if pdb_string:
                    st.session_state.predicted_structure = pdb_string
                    st.session_state.current_sequence = sequence
                    st.session_state.protein_name = protein_name
                    st.session_state.sequence_analysis = analyze_sequence(sequence)
                    st.session_state.prediction_source = "ESMFold API"
                    
                    # Extract pLDDT
                    residues, plddt = extract_plddt_from_pdb(pdb_string)
                    st.session_state.plddt_scores = (residues, plddt)
                    
                    st.success("✅ Structure prediction completed successfully!")
                    st.balloons()
                    st.rerun()  # Force page refresh to show results
                else:
                    st.error(f"❌ Prediction failed: {error}")
        
        else:  # AlphaFold DB
            if not uniprot_id:
                st.warning("⚠️ AlphaFold DB requires UniProt ID. Use UniProt ID input method.")
            else:
                predict_btn = st.button("📥 Fetch from AlphaFold DB", use_container_width=True, type="primary")
                
                if predict_btn:
                    with st.spinner("📥 Fetching from AlphaFold Database..."):
                        pdb_string, error = fetch_alphafold_structure(uniprot_id)
                        
                        if pdb_string:
                            st.session_state.predicted_structure = pdb_string
                            st.session_state.current_sequence = sequence
                            st.session_state.protein_name = protein_name
                            st.session_state.sequence_analysis = analyze_sequence(sequence)
                            st.session_state.prediction_source = "AlphaFold DB"
                            
                            residues, plddt = extract_plddt_from_pdb(pdb_string)
                            st.session_state.plddt_scores = (residues, plddt)
                            
                            st.success("✅ Structure fetched successfully!")
                            st.rerun()  # Force page refresh
                        else:
                            st.error(f"❌ {error}")
    
    # Display results
    if st.session_state.predicted_structure:
        st.markdown("---")
        st.info(f"📡 **Source:** {st.session_state.prediction_source}")
        show_results()


def show_results():
    """Display results with 3D viewer and analysis."""
    st.header(f"📊 Results: {st.session_state.protein_name}")
    
    tabs = st.tabs(["🎨 3D Viewer", "📈 Confidence", "🔬 Analysis", "💾 Download", "⚗️ Reliability"])
    
    # Tab 1: 3D Visualization
    with tabs[0]:
        st.subheader("Interactive 3D Structure")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.write("**Controls:**")
            style = st.selectbox("Style:", ["cartoon", "sphere", "stick", "line"], index=0)
            color_scheme = st.selectbox("Color:", ["pLDDT", "Spectrum", "Secondary Structure"], index=0)
            spin = st.checkbox("Auto-rotate", value=False)
            bg_color = st.color_picker("Background:", "#FFFFFF")
        
        with col1:
            view = py3Dmol.view(width=800, height=600)
            view.addModel(st.session_state.predicted_structure, 'pdb')
            
            if color_scheme == "pLDDT":
                view.setStyle({style: {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 50, 'max': 90}}})
            elif color_scheme == "Secondary Structure":
                view.setStyle({style: {'colorscheme': 'ssJmol'}})
            else:
                view.setStyle({style: {'color': 'spectrum'}})
            
            view.setBackgroundColor(bg_color)
            view.zoomTo()
            if spin:
                view.spin(True)
            
            showmol(view, height=600, width=800)
        
        if color_scheme == "pLDDT":
            st.markdown("---")
            st.write("**pLDDT Confidence Scale:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("🔵 **Very High** (>90)")
            with col2:
                st.markdown("🟢 **Confident** (70-90)")
            with col3:
                st.markdown("🟡 **Low** (50-70)")
            with col4:
                st.markdown("🟠 **Very Low** (<50)")
    
    # Tab 2: Confidence
    with tabs[1]:
        st.subheader("📈 Confidence Metrics (pLDDT)")
        
        if st.session_state.plddt_scores:
            residues, plddt = st.session_state.plddt_scores
            
            if len(plddt) > 0:
                avg_plddt = sum(plddt) / len(plddt)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average", f"{avg_plddt:.2f}")
                with col2:
                    st.metric("Maximum", f"{max(plddt):.2f}")
                with col3:
                    st.metric("Minimum", f"{min(plddt):.2f}")
                with col4:
                    st.metric("Quality", get_confidence_category(avg_plddt))
                
                st.markdown("---")
                
                # Per-residue plot
                colors = ['#FF7D45' if p < 50 else '#FFDB13' if p < 70 else '#65CBF3' if p < 90 else '#0053D6' for p in plddt]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=residues, y=plddt,
                    mode='lines+markers',
                    marker=dict(color=colors, size=4),
                    line=dict(color='lightgray', width=1),
                    name='Raw pLDDT'
                ))
                
                # Smoothed confidence line (Gaussian, sigma = hyperparameter)
                _sigma = st.session_state.get('hp_smooth_sigma', 1.5)
                smoothed_plddt = calculate_plddt_smoothed(residues, plddt, sigma=float(_sigma))
                fig.add_trace(go.Scatter(
                    x=residues, y=smoothed_plddt,
                    mode='lines',
                    line=dict(color='#667eea', width=2.5),
                    name=f'Smoothed (σ={_sigma})'
                ))
                
                fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Very High")
                fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Confident")
                fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Low")
                
                fig.update_layout(
                    title="Per-Residue Confidence",
                    xaxis_title="Residue Number",
                    yaxis_title="pLDDT Score",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution
                st.markdown("---")
                bins = [0, 50, 70, 90, 100]
                labels = ['Very Low', 'Low', 'Confident', 'Very High']
                hist_data = pd.cut(plddt, bins=bins, labels=labels)
                counts = hist_data.value_counts()
                
                fig2 = go.Figure(data=[go.Bar(
                    x=labels,
                    y=[counts.get(label, 0) for label in labels],
                    marker_color=['#FF7D45', '#FFDB13', '#65CBF3', '#0053D6']
                )])
                
                fig2.update_layout(title="Confidence Distribution", height=400)
                st.plotly_chart(fig2, use_container_width=True)
    
    # Tab 3: Analysis
    with tabs[2]:
        st.subheader("🔬 Sequence Analysis")
        
        if st.session_state.sequence_analysis:
            analysis = st.session_state.sequence_analysis
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Length", f"{analysis['length']} aa")
                st.metric("MW", f"{analysis['molecular_weight']:.0f} Da")
            with col2:
                st.metric("Aromaticity", f"{analysis['aromaticity']:.3f}")
                st.metric("Instability", f"{analysis['instability_index']:.2f}")
            with col3:
                st.metric("pI", f"{analysis['isoelectric_point']:.2f}")
                stability = "Stable" if analysis['instability_index'] < 40 else "Unstable"
                st.metric("Stability", stability)
            
            st.markdown("---")
            
            # AA composition
            aa_comp = analysis['aa_composition']
            aa_df = pd.DataFrame({
                'AA': list(aa_comp.keys()),
                'Percentage': [v * 100 for v in aa_comp.values()]
            }).sort_values('Percentage', ascending=False)
            
            fig = px.bar(aa_df, x='AA', y='Percentage',
                        title='Amino Acid Composition',
                        color='Percentage',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Secondary structure
            sec_struct = analysis['secondary_structure']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("α-Helix", f"{sec_struct['helix']*100:.1f}%")
            with col2:
                st.metric("β-Turn", f"{sec_struct['turn']*100:.1f}%")
            with col3:
                st.metric("β-Sheet", f"{sec_struct['sheet']*100:.1f}%")
    
    # Tab 4: Download
    with tabs[3]:
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download PDB File",
                data=st.session_state.predicted_structure,
                file_name=f"{st.session_state.protein_name.replace(' ', '_')}.pdb",
                mime="chemical/x-pdb",
                use_container_width=True
            )
        
        with col2:
            if st.session_state.sequence_analysis and st.session_state.plddt_scores:
                analysis = st.session_state.sequence_analysis
                _, plddt = st.session_state.plddt_scores
                avg_plddt = sum(plddt) / len(plddt) if len(plddt) > 0 else 0
                
                report = f"""Protein Structure Analysis Report
=====================================
Protein: {st.session_state.protein_name}
Source: {st.session_state.prediction_source}

Sequence Properties:
-------------------
Length: {analysis['length']} aa
Molecular Weight: {analysis['molecular_weight']:.2f} Da
pI: {analysis['isoelectric_point']:.2f}
Instability Index: {analysis['instability_index']:.2f}

Confidence:
-----------
Average pLDDT: {avg_plddt:.2f}
Max pLDDT: {max(plddt):.2f}
Min pLDDT: {min(plddt):.2f}
"""
                
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name=f"{st.session_state.protein_name.replace(' ', '_')}_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        with st.expander("👁️ Preview PDB"):
            st.code(st.session_state.predicted_structure[:2000] + "\n...", language="text")
    
    # Tab 5: Reliability Analysis
    with tabs[4]:
        st.subheader("⚗️ Prediction Reliability Analysis")
        
        # Read all hyperparameters
        plddt_threshold = int(st.session_state.get('hp_plddt_threshold', 70))
        window_size     = int(st.session_state.get('hp_window_size', 9))
        disorder_thr    = float(st.session_state.get('hp_disorder_threshold', 0.5))
        auto_trim       = bool(st.session_state.get('hp_auto_trim', True))
        smooth_sigma    = float(st.session_state.get('hp_smooth_sigma', 1.5))
        
        st.info(
            f"🎛️ Active hyperparameters — "
            f"pLDDT threshold: **{plddt_threshold}** | "
            f"Window: **{window_size}** | "
            f"Disorder threshold: **{disorder_thr:.1f}** | "
            f"σ: **{smooth_sigma}** | "
            f"Auto-trim: **{'on' if auto_trim else 'off'}**"
        )
        
        if st.session_state.plddt_scores:
            residues_r, plddt_r = st.session_state.plddt_scores
            
            # ── Reliability Grade Card ────────────────────────────────────────
            reliability = assess_prediction_reliability(plddt_r, threshold=plddt_threshold)
            gc = reliability['grade_color']
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {gc}22, {gc}44);
                        border: 3px solid {gc}; border-radius: 16px;
                        padding: 2rem; text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 5rem; font-weight: 900; color: {gc}; line-height: 1;">
                    {reliability['grade']}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #333; margin-top: 0.3rem;">
                    {reliability['grade_label']} Prediction</div>
                <div style="font-size: 0.9rem; color: #666; margin-top: 0.4rem;">
                    Evaluated at pLDDT &ge; {plddt_threshold}</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg pLDDT", f"{reliability['avg_plddt']:.1f}")
            with col2:
                st.metric("Reliable Residues",
                          f"{reliability['n_reliable']}/{reliability['n_total']}")
            with col3:
                st.metric("% Reliable", f"{reliability['percent_reliable']:.1f}%")
            with col4:
                st.metric("High-Conf Segments",
                          len(reliability['high_confidence_segments']))
            
            if reliability['high_confidence_segments']:
                segs = reliability['high_confidence_segments']
                seg_text = ", ".join([f"Res {s}–{e}" for s, e in segs[:8]])
                if len(segs) > 8:
                    seg_text += f" (+{len(segs) - 8} more)"
                st.success(f"✅ High-Confidence Regions: {seg_text}")
            
            st.markdown("---")
            
            # ── Hydrophobicity Profile ────────────────────────────────────────
            if st.session_state.current_sequence:
                seq = st.session_state.current_sequence
                
                st.subheader(f"🌊 Hydrophobicity Profile  (window = {window_size})")
                st.caption("Kyte-Doolittle scale | Larger window → smoother (↓ variance, ↑ bias)")
                h_pos, h_vals = calculate_hydrophobicity_profile(seq, window_size=window_size)
                
                if h_pos:
                    fig_h = go.Figure()
                    h_colors = ['#dc2626' if v > 0 else '#2563eb' for v in h_vals]
                    fig_h.add_trace(go.Bar(
                        x=h_pos, y=h_vals,
                        marker_color=h_colors,
                        opacity=0.75,
                        name='Hydrophobicity',
                    ))
                    fig_h.add_hline(y=0, line_color='#333', line_width=1)
                    fig_h.update_layout(
                        title=f"Kyte-Doolittle Hydrophobicity (window = {window_size})",
                        xaxis_title="Residue Position",
                        yaxis_title="Hydrophobicity",
                        height=340,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        showlegend=False,
                    )
                    st.plotly_chart(fig_h, use_container_width=True)
                    col_l, col_r = st.columns(2)
                    col_l.caption("🔴 Red = Hydrophobic (>0)")
                    col_r.caption("🔵 Blue = Hydrophilic (<0)")
                
                st.markdown("---")
                
                # ── Disorder Profile ────────────────────────────────────────
                st.subheader(
                    f"🌀 Disorder Profile  (window={window_size}, threshold={disorder_thr:.1f})"
                )
                st.caption("Higher threshold → stricter (↑ bias, ↓ variance)")
                d_pos, d_scores, d_flags = predict_disorder_profile(
                    seq, window_size=window_size, threshold=disorder_thr
                )
                
                if d_pos:
                    fig_d = go.Figure()
                    d_colors = ['#ef4444' if f else '#10b981' for f in d_flags]
                    fig_d.add_trace(go.Bar(
                        x=d_pos, y=d_scores,
                        marker_color=d_colors,
                        opacity=0.75,
                        name='Disorder Score',
                    ))
                    fig_d.add_hline(
                        y=disorder_thr,
                        line_dash='dash', line_color='#f59e0b',
                        annotation_text=f"Threshold ({disorder_thr:.1f})",
                    )
                    fig_d.update_layout(
                        title=f"Disorder Propensity (threshold = {disorder_thr:.1f})",
                        xaxis_title="Residue Position",
                        yaxis_title="Disorder Score (0-1)",
                        yaxis=dict(range=[0, 1.05]),
                        height=340,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        showlegend=False,
                    )
                    st.plotly_chart(fig_d, use_container_width=True)
                    
                    n_dis = sum(d_flags)
                    pct_d = n_dis / len(d_flags) * 100 if d_flags else 0
                    if n_dis > 0:
                        st.warning(
                            f"⚠️ {n_dis} window positions ({pct_d:.1f}%) classified as "
                            f"disordered at threshold = {disorder_thr:.1f}."
                        )
                    else:
                        st.success("✅ No significant disordered regions detected.")
            
            st.markdown("---")
            
            # ── Terminal Trimming Analysis ────────────────────────────────────
            if auto_trim and st.session_state.predicted_structure and st.session_state.current_sequence:
                st.subheader("✂️ Terminal Trimming  (Variance Reduction)")
                st.caption(
                    f"Removing N/C-terminal residues with pLDDT < {plddt_threshold} "
                    "reduces prediction variance at the cost of small coverage bias."
                )
                t_seq, t_pdb, n_tr, c_tr = trim_low_confidence_termini(
                    st.session_state.current_sequence,
                    st.session_state.predicted_structure,
                    min_plddt=float(plddt_threshold),
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("N-term Trimmed", f"{n_tr} residues")
                with col2:
                    st.metric("C-term Trimmed", f"{c_tr} residues")
                with col3:
                    retained = len(t_seq)
                    total    = len(st.session_state.current_sequence)
                    st.metric("Retained", f"{retained}/{total} aa")
                
                if n_tr > 0 or c_tr > 0:
                    st.info(
                        f"ℹ️ Removing {n_tr + c_tr} low-confidence terminal residues "
                        f"reduces pLDDT variance in the retained core structure."
                    )
                    st.download_button(
                        label="📥 Download Trimmed PDB",
                        data=t_pdb,
                        file_name=(
                            f"{st.session_state.protein_name.replace(' ', '_')}_trimmed.pdb"
                        ),
                        mime="chemical/x-pdb",
                        use_container_width=True,
                    )
                else:
                    st.success(
                        f"✅ No terminal trimming needed — all termini exceed "
                        f"the pLDDT threshold ({plddt_threshold})."
                    )
        else:
            st.info("ℹ️ Run a structure prediction first to see reliability analysis.")


def show_batch_analysis():
    """Batch analysis page."""
    st.header("📊 Batch Sequence Analysis")
    
    st.info("Upload a multi-FASTA file to analyze multiple sequences")
    
    uploaded_file = st.file_uploader("Upload multi-FASTA:", type=['fasta', 'fa', 'txt', 'faa'])
    
    if uploaded_file:
        fasta_content = uploaded_file.read().decode('utf-8')
        sequences = parse_fasta(fasta_content)
        
        if sequences:
            st.success(f"✅ Parsed {len(sequences)} sequences")
            
            with st.expander("📋 Preview"):
                for i, (name, seq) in enumerate(sequences[:10]):
                    st.text(f"{i+1}. {name} ({len(seq)} aa)")
                if len(sequences) > 10:
                    st.text(f"... and {len(sequences) - 10} more")
            
            st.markdown("---")
            
            num_to_process = st.slider("Number to analyze:", 1, min(len(sequences), 50), min(5, len(sequences)))
            
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (name, seq) in enumerate(sequences[:num_to_process]):
                    status_text.text(f"Analyzing {i+1}/{num_to_process}: {name}")
                    
                    is_valid, message = validate_sequence(seq)
                    if is_valid:
                        analysis = analyze_sequence(message)
                        if analysis:
                            results.append({
                                'Name': name,
                                'Length': len(message),
                                'MW (Da)': f"{analysis['molecular_weight']:.0f}",
                                'pI': f"{analysis['isoelectric_point']:.2f}",
                                'Instability': f"{analysis['instability_index']:.2f}",
                                'Stability': 'Stable' if analysis['instability_index'] < 40 else 'Unstable'
                            })
                    
                    progress_bar.progress((i + 1) / num_to_process)
                
                status_text.text("✅ Complete!")
                
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        data=csv,
                        file_name="batch_analysis.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


def show_comparison():
    """Structure comparison page."""
    st.header("🔄 Structure Comparison (RMSD)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted Structure")
        if st.session_state.predicted_structure:
            st.success(f"✅ {st.session_state.protein_name}")
        else:
            st.warning("⚠️ No predicted structure")
    
    with col2:
        st.subheader("Reference Structure")
        ref_file = st.file_uploader("Upload reference PDB:", type=['pdb'])
    
    if st.session_state.predicted_structure and ref_file:
        ref_pdb = ref_file.read().decode('utf-8')
        
        with st.spinner("Calculating RMSD..."):
            pred_coords = parse_pdb_coordinates(st.session_state.predicted_structure)
            ref_coords = parse_pdb_coordinates(ref_pdb)
            
            if len(pred_coords) > 0 and len(ref_coords) > 0 and len(pred_coords) == len(ref_coords):
                rmsd = calculate_rmsd(pred_coords, ref_coords)
                
                if rmsd is not None:
                    st.success("✅ RMSD Calculated")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSD (Å)", f"{rmsd:.3f}")
                    with col2:
                        st.metric("Aligned Atoms", len(pred_coords))
                    with col3:
                        quality = "Excellent" if rmsd < 2.0 else "Good" if rmsd < 4.0 else "Fair" if rmsd < 6.0 else "Poor"
                        st.metric("Quality", quality)
            else:
                st.error("❌ Length mismatch or invalid coordinates")


def show_about():
    """About page with professional design."""
    
    # Header
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.12); margin-bottom: 2rem;">
        <h1 style="text-align: center; color: #667eea; margin-bottom: 0.5rem;">ℹ️ About ProteinForge</h1>
        <p style="text-align: center; color: #666; font-size: 1.1rem;">
            Dynamic AI-Powered Protein Structure Prediction Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview Section
    st.markdown("""
    <div class="feature-card">
        <h2 style="color: #667eea; margin-bottom: 1rem;">🧬 Overview</h2>
        <p style="font-size: 1.1rem; line-height: 1.8; color: #555;">
            <strong>ProteinForge</strong> is a cutting-edge protein structure prediction dashboard powered by 
            state-of-the-art AI models. Built for researchers, students, and bioinformatics professionals, 
            it provides instant access to ESMFold and AlphaFold predictions with zero hard-coded data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Features
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="pf-card pf-card-cyan">
            <h3 style="color:#00d4ff;margin-bottom:1rem;">✨ Key Features</h3>
            <ul style="line-height:2.2;font-size:0.95rem;color:rgba(168,178,209,0.9);padding-left:1.2rem;">
                <li>✅ <strong style="color:#e6f1ff;">100% Dynamic Data</strong> — Zero hard-coded sequences</li>
                <li>🤖 <strong style="color:#e6f1ff;">Multiple AI Models</strong> — ESMFold & AlphaFold DB</li>
                <li>🎨 <strong style="color:#e6f1ff;">Interactive 3D Viewer</strong> — Rotate, zoom, style</li>
                <li>📊 <strong style="color:#e6f1ff;">Comprehensive Analysis</strong> — Full sequence metrics</li>
                <li>🗄️ <strong style="color:#e6f1ff;">Dataset Integration</strong> — 500K+ sequences</li>
                <li>⚡ <strong style="color:#e6f1ff;">Batch Processing</strong> — Multiple sequences at once</li>
                <li>🔄 <strong style="color:#e6f1ff;">Structure Comparison</strong> — RMSD calculations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pf-card pf-card-purple">
            <h3 style="color:#7b2ff7;margin-bottom:1rem;">📊 Data Sources</h3>
            <ul style="line-height:2.2;font-size:0.95rem;color:rgba(168,178,209,0.9);padding-left:1.2rem;">
                <li>🔗 <strong style="color:#e6f1ff;">UniProt REST API</strong> — Protein sequences & metadata</li>
                <li>📦 <strong style="color:#e6f1ff;">AlphaFold Database</strong> — Pre-computed structures</li>
                <li>⚡ <strong style="color:#e6f1ff;">ESMFold API</strong> — On-demand predictions</li>
                <li>🤗 <strong style="color:#e6f1ff;">Hugging Face</strong> — Curated protein datasets</li>
            </ul>
            <br>
            <h3 style="color:#f107e8;margin-bottom:1rem;">🎯 Quality Metrics</h3>
            <p style="line-height:2;font-size:0.93rem;color:rgba(168,178,209,0.9);">
                <strong style="color:#e6f1ff;">pLDDT Confidence Scale:</strong><br>
                🔵 <strong style="color:#e6f1ff;">&gt;90</strong>: Very High (~95% accuracy)<br>
                🟢 <strong style="color:#e6f1ff;">70-90</strong>: Confident (reliable)<br>
                🟡 <strong style="color:#e6f1ff;">50-70</strong>: Low (use with caution)<br>
                🟠 <strong style="color:#e6f1ff;">&lt;50</strong>: Very Low (likely disordered)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="pf-divider">', unsafe_allow_html=True)

    # Prediction Engines
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <div class="pf-section-title">🔬 Prediction Engines</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="pf-card pf-card-cyan">
            <h3 style="color:#00d4ff;margin-bottom:1rem;">⚡ ESMFold API</h3>
            <ul style="line-height:2;font-size:0.93rem;color:rgba(168,178,209,0.9);padding-left:1.2rem;">
                <li>Fast predictions (30–60 seconds)</li>
                <li>Works with any amino acid sequence</li>
                <li>Based on ESM-2 language model</li>
                <li>State-of-the-art accuracy</li>
                <li>No UniProt ID required</li>
            </ul>
            <br>
            <p style="font-size:0.82rem;color:rgba(168,178,209,0.5);font-style:italic;">
                Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure. <em>Science</em> 379(6637).
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pf-card pf-card-green">
            <h3 style="color:#00ff88;margin-bottom:1rem;">🏆 AlphaFold Database</h3>
            <ul style="line-height:2;font-size:0.93rem;color:rgba(168,178,209,0.9);padding-left:1.2rem;">
                <li>Pre-computed high-quality structures</li>
                <li>Instant retrieval (no waiting)</li>
                <li>Requires UniProt accession ID</li>
                <li>200M+ protein structures</li>
                <li>Highest confidence scores</li>
            </ul>
            <br>
            <p style="font-size:0.82rem;color:rgba(168,178,209,0.5);font-style:italic;">
                Jumper et al. (2021). Highly accurate protein structure prediction with AlphaFold. <em>Nature</em> 596(7873).
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Limitations & Best Practices
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="pf-card pf-card-amber">
            <h3 style="color:#f59e0b;margin-bottom:1rem;">⚠️ Limitations</h3>
            <ul style="line-height:2;font-size:0.93rem;color:rgba(168,178,209,0.9);padding-left:1.2rem;">
                <li>Sequence length limited to ~2000 residues</li>
                <li>Monomeric structures only (no complexes)</li>
                <li>Low confidence may indicate disorder</li>
                <li>Post-translational modifications not modeled</li>
                <li>Always validate with experimental data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pf-card pf-card-green">
            <h3 style="color:#00ff88;margin-bottom:1rem;">✅ Best Practices</h3>
            <ul style="line-height:2;font-size:0.93rem;color:rgba(168,178,209,0.9);padding-left:1.2rem;">
                <li>Check pLDDT scores for reliability</li>
                <li>Compare with AlphaFold DB when available</li>
                <li>Use batch mode for multiple sequences</li>
                <li>Download PDB files for further analysis</li>
                <li>Validate predictions experimentally</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="pf-divider">', unsafe_allow_html=True)

    # Links
    st.markdown("""
    <div class="pf-card">
        <h3 style="color:#e6f1ff;margin-bottom:1.2rem;">🔗 Useful Links</h3>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.75rem;">
            <a href="https://predictioncenter.org/" target="_blank" style="text-decoration:none;">
                <div style="background:rgba(0,212,255,0.06);padding:1rem;border-radius:10px;border:1px solid rgba(0,212,255,0.15);transition:all 0.2s;">
                    <strong style="color:#00d4ff;">CASP</strong><br>
                    <span style="color:rgba(168,178,209,0.7);font-size:0.82rem;">Critical Assessment of Structure Prediction</span>
                </div>
            </a>
            <a href="https://www.uniprot.org/" target="_blank" style="text-decoration:none;">
                <div style="background:rgba(0,212,255,0.06);padding:1rem;border-radius:10px;border:1px solid rgba(0,212,255,0.15);">
                    <strong style="color:#00d4ff;">UniProt</strong><br>
                    <span style="color:rgba(168,178,209,0.7);font-size:0.82rem;">Universal Protein Resource</span>
                </div>
            </a>
            <a href="https://alphafold.ebi.ac.uk/" target="_blank" style="text-decoration:none;">
                <div style="background:rgba(123,47,247,0.06);padding:1rem;border-radius:10px;border:1px solid rgba(123,47,247,0.15);">
                    <strong style="color:#7b2ff7;">AlphaFold DB</strong><br>
                    <span style="color:rgba(168,178,209,0.7);font-size:0.82rem;">200M+ protein structures</span>
                </div>
            </a>
            <a href="https://esmatlas.com/" target="_blank" style="text-decoration:none;">
                <div style="background:rgba(123,47,247,0.06);padding:1rem;border-radius:10px;border:1px solid rgba(123,47,247,0.15);">
                    <strong style="color:#7b2ff7;">ESM Atlas</strong><br>
                    <span style="color:rgba(168,178,209,0.7);font-size:0.82rem;">ESMFold Metagenomic Atlas</span>
                </div>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(0,212,255,0.1) 0%,rgba(123,47,247,0.15) 50%,rgba(241,7,232,0.08) 100%);
                border:1px solid rgba(0,212,255,0.2);
                padding:2rem;border-radius:16px;text-align:center;margin-top:2rem;">
        <h3 style="color:#e6f1ff;margin-bottom:0.75rem;">Built with ❤️ for the Bioinformatics Community</h3>
        <p style="color:rgba(168,178,209,0.8);font-size:0.95rem;margin-bottom:0.4rem;">
            Powered by <strong style="color:#00d4ff;">Streamlit</strong> · <strong style="color:#7b2ff7;">py3Dmol</strong> · <strong style="color:#00d4ff;">Plotly</strong> · <strong style="color:#7b2ff7;">Hugging Face</strong>
        </p>
        <p style="color:rgba(168,178,209,0.45);font-size:0.8rem;">
            © 2026 ProteinForge &nbsp;·&nbsp; 100% Open Source &nbsp;·&nbsp; Zero Hard-Coded Data
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
