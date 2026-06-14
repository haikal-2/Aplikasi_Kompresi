import streamlit as st
import io
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

# ============================================================
# FUNGSI KOMPRESI
# ============================================================

def compress_jpeg_standard(image_pil, quality_value):
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG", quality=quality_value)
    buffer.seek(0)
    img_bytes = buffer.getvalue()
    compressed_img = Image.open(buffer)
    return compressed_img, len(img_bytes), img_bytes

def compress_kmeans(image_pil, n_colors):
    img_np = np.array(image_pil)
    h, w, c = img_np.shape
    pixels = img_np.reshape(-1, c)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=3)
    labels = kmeans.fit_predict(pixels)
    colors = kmeans.cluster_centers_.astype(np.uint8)
    compressed_pixels = colors[labels]
    compressed_np = compressed_pixels.reshape(h, w, c)
    compressed_img = Image.fromarray(compressed_np)
    buffer = io.BytesIO()
    compressed_img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    return compressed_img, len(img_bytes), img_bytes

def compress_svd(image_pil, k_values):
    img_np = np.array(image_pil)
    compressed_channels = []
    for i in range(3):
        channel = img_np[:, :, i].astype(float)
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        compressed_channel = np.dot(U[:, :k_values], np.dot(np.diag(S[:k_values]), Vt[:k_values, :]))
        compressed_channels.append(compressed_channel)
    compressed_np = np.stack(compressed_channels, axis=2)
    compressed_np = np.clip(compressed_np, 0, 255).astype(np.uint8)
    compressed_img = Image.fromarray(compressed_np)
    buffer = io.BytesIO()
    compressed_img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    return compressed_img, len(img_bytes), img_bytes

def compress_combined(image_pil, quality_value, n_colors, k_values):
    # Step 1: K-Means
    img_np = np.array(image_pil)
    h, w, c = img_np.shape
    pixels = img_np.reshape(-1, c)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=3)
    labels = kmeans.fit_predict(pixels)
    colors = kmeans.cluster_centers_.astype(np.uint8)
    img_k = Image.fromarray(colors[labels].reshape(h, w, c))

    # Step 2: SVD
    img_np_2 = np.array(img_k)
    compressed_channels = []
    for i in range(3):
        channel = img_np_2[:, :, i].astype(float)
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        compressed_channel = np.dot(U[:, :k_values], np.dot(np.diag(S[:k_values]), Vt[:k_values, :]))
        compressed_channels.append(compressed_channel)
    img_svd = Image.fromarray(np.clip(np.stack(compressed_channels, axis=2), 0, 255).astype(np.uint8))

    # Step 3: JPEG
    buffer = io.BytesIO()
    img_svd.save(buffer, format="JPEG", quality=quality_value)
    buffer.seek(0)
    img_bytes = buffer.getvalue()
    return Image.open(buffer), len(img_bytes), img_bytes

def calculate_psnr(original, compressed):
    orig_np = np.array(original).astype(float)
    comp_np = np.array(compressed).astype(float)
    mse = np.mean((orig_np - comp_np) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Komparasi Kompresi Gambar",
    layout="wide",
    page_icon="K4"
)

# ============================================================
# CUSTOM CSS — Dark Research Dashboard Theme
# ============================================================
st.markdown("""
<style>
    /* ──────── BASE ──────── */
    html, body, .stApp {
        background-color: #0a0e17 !important;
        color: #cdd6f4 !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1400px; }

    /* ──────── HERO HEADER ──────── */
    .hero {
        background: linear-gradient(160deg, #0f1b35 0%, #0d1b2a 40%, #110d20 100%);
        border: 1px solid #1e2d45;
        border-radius: 20px;
        padding: 2.8rem 2.5rem 2.4rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(96,165,250,0.07) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f0f4ff;
        margin: 0 0 0.5rem;
        letter-spacing: -0.5px;
    }
    .hero-title em { font-style: normal; color: #60a5fa; }
    .hero-sub { color: #7a8ba8; font-size: 0.95rem; margin: 0 0 1.4rem; }
    .badge-row { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
    .badge {
        font-size: 0.76rem; font-weight: 700;
        padding: 4px 14px; border-radius: 20px;
        letter-spacing: 0.3px;
    }
    .b-jpeg  { background: #0c1f3a; border: 1px solid #1d4ed8; color: #60a5fa; }
    .b-km    { background: #0b2014; border: 1px solid #15803d; color: #4ade80; }
    .b-svd   { background: #1a0d2e; border: 1px solid #7c3aed; color: #c084fc; }
    .b-hyb   { background: #2a0f0f; border: 1px solid #b91c1c; color: #f87171; }

    /* ──────── SETTINGS PANEL ──────── */
    .param-card {
        background: #111827;
        border: 1px solid #1f2d3d;
        border-radius: 12px;
        padding: 1rem 1.2rem 0.6rem;
        border-top: 3px solid;
        margin-bottom: 0.25rem;
    }
    .param-card.c-jpeg  { border-top-color: #3b82f6; }
    .param-card.c-km    { border-top-color: #22c55e; }
    .param-card.c-svd   { border-top-color: #a855f7; }
    .param-card h4 { margin: 0 0 2px; font-size: 0.88rem; font-weight: 700; }
    .param-card p  { color: #6b7280; font-size: 0.76rem; margin: 0 0 0.6rem; }

    /* ──────── UPLOAD STATS ──────── */
    .stat-strip {
        display: flex; gap: 12px; margin: 1rem 0 0;
    }
    .stat-pill {
        background: #111827; border: 1px solid #1f2d3d;
        border-radius: 10px; padding: 0.6rem 1.1rem;
        flex: 1; text-align: center;
    }
    .stat-pill .val { font-size: 1.35rem; font-weight: 800; color: #f0f4ff; }
    .stat-pill .lbl { font-size: 0.72rem; color: #6b7280; letter-spacing: 0.5px; }

    /* ──────── RUN BUTTON ──────── */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #3b82f6 100%) !important;
        color: #fff !important; border: none !important;
        border-radius: 12px !important; font-weight: 700 !important;
        font-size: 1rem !important; letter-spacing: 0.4px !important;
        padding: 0.7rem 2rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(37,99,235,0.25) !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59,130,246,0.4) !important;
    }

    /* ──────── DOWNLOAD BUTTON ──────── */
    div[data-testid="stDownloadButton"] > button {
        background: #111827 !important; color: #9ca3af !important;
        border: 1px solid #1f2937 !important; border-radius: 8px !important;
        font-size: 0.78rem !important; width: 100% !important;
        padding: 0.35rem 0 !important; transition: all 0.15s !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: #1f2937 !important; color: #e2e8f0 !important;
        border-color: #3b82f6 !important;
    }

    /* ──────── METRIC WIDGET ──────── */
    div[data-testid="metric-container"] {
        background: #111827; border: 1px solid #1f2d3d;
        border-radius: 10px; padding: 1rem 1.2rem;
    }
    div[data-testid="metric-container"] label { color: #6b7280 !important; font-size: 0.78rem !important; }
    div[data-testid="stMetricValue"] { color: #f0f4ff !important; font-size: 1.6rem !important; font-weight: 800 !important; }
    div[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

    /* ──────── FILE DIVIDER ──────── */
    .file-banner {
        background: #111827;
        border: 1px solid #1f2d3d;
        border-left: 4px solid #3b82f6;
        border-radius: 0 12px 12px 0;
        padding: 0.8rem 1.4rem;
        margin: 2rem 0 1.2rem;
        display: flex; align-items: center; gap: 0.75rem;
    }
    .file-num {
        background: #0c1f3a; border: 1px solid #1d4ed8;
        color: #60a5fa; font-weight: 800;
        font-size: 0.8rem; padding: 2px 10px; border-radius: 6px;
    }
    .file-name { color: #e2e8f0; font-size: 0.9rem; font-weight: 700; }
    .file-meta { color: #4b5563; font-size: 0.78rem; margin-left: auto; font-family: monospace; }

    /* ──────── IMAGE COLUMN LABELS ──────── */
    .col-label {
        font-size: 0.7rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: 1.5px;
        padding: 4px 0; border-radius: 6px;
        text-align: center; margin-bottom: 6px;
        display: block;
    }
    .cl-before { color: #9ca3af; border-bottom: 2px solid #374151; }
    .cl-jpeg   { color: #60a5fa; border-bottom: 2px solid #1d4ed8; }
    .cl-km     { color: #4ade80; border-bottom: 2px solid #15803d; }
    .cl-svd    { color: #c084fc; border-bottom: 2px solid #7c3aed; }
    .cl-hyb    { color: #f87171; border-bottom: 2px solid #b91c1c; }

    /* ──────── STAT BOX (below each image) ──────── */
    .stat-box {
        background: #0f1623; border: 1px solid #1a2535;
        border-radius: 8px; padding: 8px 10px; margin-top: 6px;
    }
    .sr { display: flex; justify-content: space-between; padding: 3px 0; font-size: 0.77rem; }
    .sk { color: #4b5563; }
    .sv { color: #e2e8f0; font-weight: 700; font-family: monospace; }
    .sg { color: #22c55e !important; font-weight: 700; font-family: monospace; }
    .sr_ { color: #ef4444 !important; font-weight: 700; font-family: monospace; }
    .sm { color: #a78bfa !important; font-weight: 700; font-family: monospace; }

    /* ──────── TABS ──────── */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827; border-radius: 10px; padding: 4px;
        border: 1px solid #1f2937; gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px;
        color: #6b7280; font-weight: 600; font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1e3a5f !important; color: #60a5fa !important;
    }

    /* ──────── EXPANDER ──────── */
    .streamlit-expanderHeader {
        background: #111827 !important; border: 1px solid #1f2937 !important;
        border-radius: 10px !important; color: #e2e8f0 !important; font-weight: 700 !important;
    }

    /* ──────── DATAFRAME ──────── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    iframe[title="st.dataframe"] { border-radius: 10px !important; }

    /* ──────── DIVIDER ──────── */
    hr { border-color: #1f2937 !important; margin: 2rem 0 !important; }

    /* ──────── EMPTY STATE ──────── */
    .empty-state {
        background: #111827; border: 2px dashed #1f2937;
        border-radius: 20px; padding: 4rem 2rem;
        text-align: center; margin-top: 2rem;
    }
    .empty-icon { font-size: 3.5rem; }
    .empty-title { color: #60a5fa; font-size: 1.4rem; font-weight: 800; margin: 1rem 0 0.4rem; }
    .empty-sub { color: #4b5563; font-size: 0.88rem; }
    .algo-chip-row { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 1.2rem; }
    .algo-chip {
        font-size: 0.78rem; font-weight: 700;
        padding: 5px 14px; border-radius: 20px;
    }

    /* ──────── PROGRESS BAR (compression ratio) ──────── */
    .ratio-bar-wrap { margin-top: 5px; }
    .ratio-bar-bg { background: #1f2937; border-radius: 4px; height: 5px; overflow: hidden; }
    .ratio-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
    .rf-jpeg { background: #3b82f6; }
    .rf-km   { background: #22c55e; }
    .rf-svd  { background: #a855f7; }
    .rf-hyb  { background: #ef4444; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">Komparasi <em>Algoritma Kompresi</em> Gambar</div>
    <p class="hero-sub">Analisis visual & numerik <b>sebelum → sesudah</b> kompresi secara side-by-side</p>
    <div class="badge-row">
        <span class="badge b-jpeg">JPEG Standard</span>
        <span class="badge b-km">K-Means</span>
        <span class="badge b-svd">SVD</span>
        <span class="badge b-hyb">Hybrid (3-in-1)</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PENGATURAN PARAMETER — TANPA SIDEBAR
# ============================================================
with st.expander("Atur Parameter Algoritma", expanded=True):
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="param-card c-jpeg">
            <h4 style="color:#60a5fa;">JPEG Standard</h4>
            <p>Kompresi lossy berbasis DCT — standar industri</p>
        </div>
        """, unsafe_allow_html=True)
        jpeg_q = st.slider("Kualitas (%)", 1, 100, 30, key="jpeg_q",
                           help="Rendah = file kecil, kualitas turun")

    with c2:
        st.markdown("""
        <div class="param-card c-km">
            <h4 style="color:#4ade80;">K-Means Clustering</h4>
            <p>Kuantisasi warna piksel ke k warna dominan</p>
        </div>
        """, unsafe_allow_html=True)
        kmeans_c = st.slider("Jumlah Warna Dominan", 2, 64, 16, key="kmeans_c",
                             help="Sedikit warna = lebih terkompresi")

    with c3:
        st.markdown("""
        <div class="param-card c-svd">
            <h4 style="color:#c084fc;">SVD</h4>
            <p>Aproksimasi matriks dengan k nilai singular terbesar</p>
        </div>
        """, unsafe_allow_html=True)
        svd_k = st.slider("Komponen Utama (k)", 10, 200, 50, key="svd_k",
                          help="Kecil k = aproksimasi lebih kasar")
    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# UPLOAD
# ============================================================
st.markdown("### Unggah Gambar")
uploaded_files = st.file_uploader(
    "Pilih gambar JPEG",
    type=["jpg", "jpeg", "JPG", "JPEG"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:
    total_kb = sum(len(f.getvalue()) for f in uploaded_files) / 1024
    ready_txt = "Siap diproses" if len(uploaded_files) >= 1 else "Min. 1 gambar"
    st.markdown(f"""
    <div class="stat-strip">
        <div class="stat-pill"><div class="val">{len(uploaded_files)}</div><div class="lbl">GAMBAR DIUNGGAH</div></div>
        <div class="stat-pill"><div class="val">{total_kb:.1f} KB</div><div class="lbl">TOTAL UKURAN ASLI</div></div>
        <div class="stat-pill"><div class="val">{ready_txt}</div><div class="lbl">STATUS</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    bc1, bc2, bc3 = st.columns([1.5, 2, 1.5])
    with bc2:
        start_btn = st.button("Mulai Komparasi", use_container_width=True)

    # ============================================================
    # PROCESSING LOOP
    # ============================================================
    if start_btn:
        report_data = []

        for index, uploaded_file in enumerate(uploaded_files):

            with st.spinner(f"Memproses gambar {index+1}/{len(uploaded_files)}: {uploaded_file.name}"):
                original_img = Image.open(uploaded_file).convert("RGB")
                size_ori = len(uploaded_file.getvalue())
                w_px, h_px = original_img.size

                img_jpeg, size_jpeg, byte_jpeg = compress_jpeg_standard(original_img, jpeg_q)
                img_km,   size_km,   byte_km   = compress_kmeans(original_img, kmeans_c)
                img_svd,  size_svd,  byte_svd  = compress_svd(original_img, svd_k)
                img_hyb,  size_hyb,  byte_hyb  = compress_combined(original_img, jpeg_q, kmeans_c, svd_k)

                psnr_jpeg = calculate_psnr(original_img, img_jpeg)
                psnr_km   = calculate_psnr(original_img, img_km)
                psnr_svd  = calculate_psnr(original_img, img_svd)
                psnr_hyb  = calculate_psnr(original_img, img_hyb)

            # ----- File banner -----
            st.markdown(f"""
            <div class="file-banner">
                <span class="file-num">#{index+1}</span>
                <span class="file-name">{uploaded_file.name}</span>
                <span class="file-meta">{w_px}×{h_px}px · {size_ori/1024:.1f} KB asli</span>
            </div>
            """, unsafe_allow_html=True)

            # ----- 5-column comparison -----
            cols = st.columns(5)

            # helper to compute & color-code susut
            def susut(s_ori, s_comp):
                pct = ((s_ori - s_comp) / s_ori) * 100
                css = "sg" if pct > 0 else "sr_"
                return pct, css

            # --- BEFORE (Original) ---
            with cols[0]:
                st.markdown('<span class="col-label cl-before">BEFORE (Asli)</span>',
                            unsafe_allow_html=True)
                st.image(original_img, use_column_width=True)
                st.markdown(f"""
                <div class="stat-box">
                    <div class="sr"><span class="sk">Ukuran</span><span class="sv">{size_ori/1024:.1f} KB</span></div>
                    <div class="sr"><span class="sk">Resolusi</span><span class="sv">{w_px}×{h_px}</span></div>
                    <div class="sr"><span class="sk">Mode</span><span class="sv">RGB</span></div>
                </div>
                """, unsafe_allow_html=True)

            # --- JPEG ---
            with cols[1]:
                pct_j, css_j = susut(size_ori, size_jpeg)
                bar_w = max(2, int(abs(pct_j)))
                st.markdown('<span class="col-label cl-jpeg">JPEG Standard</span>',
                            unsafe_allow_html=True)
                st.image(img_jpeg, use_column_width=True)
                st.markdown(f"""
                <div class="stat-box">
                    <div class="sr"><span class="sk">Ukuran</span><span class="sv">{size_jpeg/1024:.1f} KB</span></div>
                    <div class="sr"><span class="sk">Susut</span><span class="{css_j}">{pct_j:+.1f}%</span></div>
                    <div class="sr"><span class="sk">PSNR</span><span class="sm">{psnr_jpeg:.1f} dB</span></div>
                    <div class="ratio-bar-wrap">
                        <div class="ratio-bar-bg"><div class="ratio-bar-fill rf-jpeg" style="width:{min(bar_w,100)}%"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.download_button("Unduh JPEG", byte_jpeg,
                                   f"JPEG_{uploaded_file.name}", "image/jpeg",
                                   key=f"d1_{index}", use_container_width=True)

            # --- K-MEANS ---
            with cols[2]:
                pct_k, css_k = susut(size_ori, size_km)
                bar_w = max(2, int(abs(pct_k)))
                st.markdown('<span class="col-label cl-km">K-Means</span>',
                            unsafe_allow_html=True)
                st.image(img_km, use_column_width=True)
                st.markdown(f"""
                <div class="stat-box">
                    <div class="sr"><span class="sk">Ukuran</span><span class="sv">{size_km/1024:.1f} KB</span></div>
                    <div class="sr"><span class="sk">Susut</span><span class="{css_k}">{pct_k:+.1f}%</span></div>
                    <div class="sr"><span class="sk">PSNR</span><span class="sm">{psnr_km:.1f} dB</span></div>
                    <div class="ratio-bar-wrap">
                        <div class="ratio-bar-bg"><div class="ratio-bar-fill rf-km" style="width:{min(bar_w,100)}%"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.download_button("Unduh K-Means", byte_km,
                                   f"KMEANS_{uploaded_file.name}", "image/jpeg",
                                   key=f"d2_{index}", use_container_width=True)

            # --- SVD ---
            with cols[3]:
                pct_s, css_s = susut(size_ori, size_svd)
                bar_w = max(2, int(abs(pct_s)))
                st.markdown('<span class="col-label cl-svd">SVD</span>',
                            unsafe_allow_html=True)
                st.image(img_svd, use_column_width=True)
                st.markdown(f"""
                <div class="stat-box">
                    <div class="sr"><span class="sk">Ukuran</span><span class="sv">{size_svd/1024:.1f} KB</span></div>
                    <div class="sr"><span class="sk">Susut</span><span class="{css_s}">{pct_s:+.1f}%</span></div>
                    <div class="sr"><span class="sk">PSNR</span><span class="sm">{psnr_svd:.1f} dB</span></div>
                    <div class="ratio-bar-wrap">
                        <div class="ratio-bar-bg"><div class="ratio-bar-fill rf-svd" style="width:{min(bar_w,100)}%"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.download_button("Unduh SVD", byte_svd,
                                   f"SVD_{uploaded_file.name}", "image/jpeg",
                                   key=f"d3_{index}", use_container_width=True)

            # --- HYBRID ---
            with cols[4]:
                pct_h, css_h = susut(size_ori, size_hyb)
                bar_w = max(2, int(abs(pct_h)))
                st.markdown('<span class="col-label cl-hyb">Hybrid (3-in-1)</span>',
                            unsafe_allow_html=True)
                st.image(img_hyb, use_column_width=True)
                st.markdown(f"""
                <div class="stat-box">
                    <div class="sr"><span class="sk">Ukuran</span><span class="sv">{size_hyb/1024:.1f} KB</span></div>
                    <div class="sr"><span class="sk">Susut</span><span class="{css_h}">{pct_h:+.1f}%</span></div>
                    <div class="sr"><span class="sk">PSNR</span><span class="sm">{psnr_hyb:.1f} dB</span></div>
                    <div class="ratio-bar-wrap">
                        <div class="ratio-bar-bg"><div class="ratio-bar-fill rf-hyb" style="width:{min(bar_w,100)}%"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.download_button("Unduh Hybrid", byte_hyb,
                                   f"COMBO_{uploaded_file.name}", "image/jpeg",
                                   key=f"d4_{index}", use_container_width=True)

            # Simpan laporan
            report_data.append({
                "Nama File":          uploaded_file.name,
                "Asli (KB)":          round(size_ori  / 1024, 2),
                "JPEG (KB)":          round(size_jpeg  / 1024, 2),
                "JPEG Susut (%)":     round(((size_ori - size_jpeg) / size_ori) * 100, 1),
                "JPEG PSNR (dB)":     round(psnr_jpeg, 2),
                "K-Means (KB)":       round(size_km    / 1024, 2),
                "K-Means Susut (%)":  round(((size_ori - size_km)   / size_ori) * 100, 1),
                "K-Means PSNR (dB)":  round(psnr_km,   2),
                "SVD (KB)":           round(size_svd   / 1024, 2),
                "SVD Susut (%)":      round(((size_ori - size_svd)  / size_ori) * 100, 1),
                "SVD PSNR (dB)":      round(psnr_svd,  2),
                "Hybrid (KB)":        round(size_hyb   / 1024, 2),
                "Hybrid Susut (%)":   round(((size_ori - size_hyb)  / size_ori) * 100, 1),
                "Hybrid PSNR (dB)":   round(psnr_hyb,  2),
            })

        # ============================================================
        # RINGKASAN AKHIR
        # ============================================================
        st.markdown("---")
        st.markdown("### Ringkasan Performa Keseluruhan")

        df = pd.DataFrame(report_data)

        # Avg susut per algoritma
        m1, m2, m3, m4 = st.columns(4)
        avg_data = [
            ("JPEG Standard", df["JPEG Susut (%)"].mean(),   df["JPEG PSNR (dB)"].mean()),
            ("K-Means",       df["K-Means Susut (%)"].mean(), df["K-Means PSNR (dB)"].mean()),
            ("SVD",           df["SVD Susut (%)"].mean(),     df["SVD PSNR (dB)"].mean()),
            ("Hybrid",        df["Hybrid Susut (%)"].mean(),  df["Hybrid PSNR (dB)"].mean()),
        ]
        for col, (lbl, susut_avg, psnr_avg) in zip([m1, m2, m3, m4], avg_data):
            col.metric(lbl, f"{susut_avg:.1f}% susut", f"PSNR rata-rata: {psnr_avg:.1f} dB")

        st.markdown("")

        tab_tbl, tab_chart = st.tabs(["Tabel Detail", "Grafik Ukuran File"])

        with tab_tbl:
            display_cols = [
                "Nama File", "Asli (KB)",
                "JPEG (KB)", "JPEG Susut (%)", "JPEG PSNR (dB)",
                "K-Means (KB)", "K-Means Susut (%)", "K-Means PSNR (dB)",
                "SVD (KB)", "SVD Susut (%)", "SVD PSNR (dB)",
                "Hybrid (KB)", "Hybrid Susut (%)", "Hybrid PSNR (dB)",
            ]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        with tab_chart:
            chart_df = df.set_index("Nama File")[
                ["Asli (KB)", "JPEG (KB)", "K-Means (KB)", "SVD (KB)", "Hybrid (KB)"]
            ]
            st.bar_chart(chart_df)

# ============================================================
# EMPTY STATE
# ============================================================
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon"></div>
        <div class="empty-title">Unggah Gambar untuk Mulai</div>
        <div class="empty-sub">Klik tombol <b>Browse files</b> di atas, pilih minimal 10 gambar JPEG, lalu tekan <b>Mulai Uji Komparasi</b></div>
        <div class="algo-chip-row">
            <span class="algo-chip b-jpeg">JPEG Standard</span>
            <span class="algo-chip b-km">K-Means</span>
            <span class="algo-chip b-svd">SVD</span>
            <span class="algo-chip b-hyb">Hybrid</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
