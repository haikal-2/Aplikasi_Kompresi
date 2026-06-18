"""
====================================================
Implementasi Regresi Linear Berganda (Multiple Linear Regression)
Dataset : California Housing — Synthetic (20.000 records)
Variabel X : Pendapatan_Median, Usia_Rumah, Rata_Kamar, Populasi (4 variabel)
Variabel Y : Harga_Rumah
Visualisasi: Correlation Heatmap + Scatter Plot Grid + Evaluation Plots
====================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model  import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regresi Linear Berganda",
    layout="wide",
    page_icon="📈"
)

# ─────────────────────────────────────────────────────────────
# CSS — Dark Research Dashboard
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, .stApp { background-color:#0a0e17 !important; color:#cdd6f4; font-family:'Segoe UI',sans-serif; }
  #MainMenu,footer,header { visibility:hidden; }
  .block-container { padding:1.5rem 2.5rem 3rem; max-width:1380px; }

  /* ─ HERO ─ */
  .hero {
    background:linear-gradient(160deg,#0f1b35 0%,#0d1b2a 40%,#110d20 100%);
    border:1px solid #1e2d45; border-radius:20px;
    padding:2.8rem 2rem 2.4rem; margin-bottom:2rem; text-align:center;
    position:relative; overflow:hidden;
  }
  .hero::after {
    content:''; position:absolute; inset:0;
    background:radial-gradient(ellipse 70% 50% at 50% 0%,rgba(96,165,250,.07) 0%,transparent 65%);
    pointer-events:none;
  }
  .hero-title { font-size:2.1rem; font-weight:900; color:#f0f4ff; margin:0 0 .5rem; letter-spacing:-.5px; }
  .hero-title em { font-style:normal; color:#60a5fa; }
  .hero-sub  { color:#7a8ba8; font-size:.95rem; margin:0; }
  .hero-note { color:#374151; font-size:.78rem; margin-top:.5rem; }
  .badge-row { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:1.2rem; }
  .badge { font-size:.75rem; font-weight:700; padding:4px 14px; border-radius:20px; }
  .bx1 { background:#0c1f3a; border:1px solid #1d4ed8; color:#60a5fa; }
  .bx2 { background:#0b2014; border:1px solid #15803d; color:#4ade80; }
  .bx3 { background:#1a0d2e; border:1px solid #7c3aed; color:#c084fc; }
  .bx4 { background:#2a180a; border:1px solid #b45309; color:#fbbf24; }
  .by  { background:#1b0d2e; border:1px solid #9f1239; color:#fb7185; }

  /* ─ SECTION HEADER ─ */
  .sec-hdr {
    font-size:1.1rem; font-weight:800; color:#f0f4ff;
    padding-bottom:.5rem; border-bottom:2px solid #1f2d3d;
    margin-bottom:1.4rem; letter-spacing:-.2px;
  }

  /* ─ STAT PILL ─ */
  .stat-row  { display:flex; gap:12px; margin-bottom:1.5rem; }
  .stat-pill { background:#111827; border:1px solid #1f2d3d; border-radius:12px;
               padding:.7rem 1.2rem; flex:1; text-align:center; }
  .sv  { font-size:1.7rem; font-weight:800; color:#60a5fa; }
  .sl  { font-size:.7rem;  color:#6b7280; letter-spacing:.6px; margin-top:2px; }

  /* ─ VAR TABLE ─ */
  .var-tbl  { width:100%; border-collapse:collapse; font-size:.84rem; margin-bottom:1rem; }
  .var-tbl th { background:#1e3a5f; color:#60a5fa; padding:8px 14px;
                font-weight:700; text-align:left; }
  .var-tbl td { background:#111827; color:#cdd6f4; padding:7px 14px;
                border-bottom:1px solid #1a2535; }
  .var-tbl tr:hover td { background:#141e2f; }
  .tx  { background:#0c1f3a; border:1px solid #1d4ed8; color:#60a5fa;
         padding:2px 9px; border-radius:10px; font-size:.7rem; font-weight:700; }
  .ty  { background:#1b0822; border:1px solid #9f1239; color:#fb7185;
         padding:2px 9px; border-radius:10px; font-size:.7rem; font-weight:700; }

  /* ─ EQUATION BOX ─ */
  .eq-box {
    background:#0b1d36; border:1px solid #1d4ed8;
    border-radius:14px; padding:1.3rem 1.8rem;
    font-family:'Courier New',monospace; font-size:.95rem;
    color:#93c5fd; text-align:center; line-height:2;
    margin-bottom:1.2rem;
  }
  .eq-box b { color:#bfdbfe; }

  /* ─ COEFF TABLE ─ */
  .coef-tbl { width:100%; border-collapse:collapse; font-size:.84rem; }
  .coef-tbl th { background:#111827; color:#60a5fa; padding:9px 14px;
                 font-weight:700; text-align:left; border-bottom:2px solid #1f2d3d; }
  .coef-tbl td { padding:8px 14px; border-bottom:1px solid #1a2535; color:#e2e8f0; }
  .coef-tbl tr:hover td { background:#0d1623; }
  .cp { color:#4ade80 !important; font-weight:700; font-family:monospace; }
  .cn { color:#f87171 !important; font-weight:700; font-family:monospace; }

  /* ─ METRIC CARD ─ */
  .mc { background:#111827; border:1px solid #1f2d3d; border-radius:12px;
        padding:1rem 1.2rem; border-left:4px solid; text-align:center; }
  .mc .mv { font-size:1.7rem; font-weight:800; color:#f0f4ff; }
  .mc .ml { font-size:.72rem; color:#6b7280; letter-spacing:.6px; margin-top:2px; }
  .mc .md { font-size:.78rem; color:#9ca3af; margin-top:5px; }

  /* ─ INTERPRETATION BOX ─ */
  .interp-box { background:#0b1d36; border:1px solid #1f2d3d; border-radius:10px;
                padding:1.1rem 1.4rem; font-size:.85rem; line-height:1.8; color:#c7d4eb; }
  .interp-box b { color:#93c5fd; }

  /* ─ TABS ─ */
  .stTabs [data-baseweb="tab-list"] { background:#111827; border-radius:10px;
    padding:4px; border:1px solid #1f2937; gap:2px; }
  .stTabs [data-baseweb="tab"] { background:transparent; border-radius:8px;
    color:#6b7280; font-weight:600; font-size:.85rem; }
  .stTabs [aria-selected="true"] { background:#1e3a5f !important; color:#60a5fa !important; }

  /* ─ MISC ─ */
  hr { border-color:#1f2937 !important; margin:2rem 0 !important; }
  div[data-testid="metric-container"] { background:#111827; border:1px solid #1f2d3d;
    border-radius:10px; padding:1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# GENERATE DATASET  (20.000 records — California Housing style)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)

    # ── Variabel Independen (X) ──────────────────────────────
    pendapatan = rng.lognormal(mean=1.50, sigma=0.52, size=n).clip(0.50, 15.00)
    usia_rumah = rng.uniform(1, 52, size=n)
    rata_kamar = rng.lognormal(mean=1.60, sigma=0.40, size=n).clip(1.00, 15.00)
    populasi   = rng.lognormal(mean=6.50, sigma=0.85, size=n).clip(3, 35_000)

    # ── Variabel Dependen (Y) — persamaan linear + noise ────
    noise      = rng.normal(0, 0.28, size=n)
    harga      = (
        -0.20
        + 0.55 * pendapatan          # koef positif kuat
        + 0.018 * usia_rumah         # koef positif lemah
        + 0.09  * rata_kamar         # koef positif lemah-sedang
        - 0.000035 * populasi        # koef negatif sangat kecil
        + noise
    ).clip(0.15, 5.00)

    return pd.DataFrame({
        "Pendapatan_Median": pendapatan.round(4),
        "Usia_Rumah":        usia_rumah.round(1),
        "Rata_Kamar":        rata_kamar.round(2),
        "Populasi":          populasi.round(0).astype(int),
        "Harga_Rumah":       harga.round(4),
    })


df = generate_data()

X_COLS = ["Pendapatan_Median", "Usia_Rumah", "Rata_Kamar", "Populasi"]
Y_COL  = "Harga_Rumah"

X = df[X_COLS]
y = df[Y_COL]

# ─────────────────────────────────────────────────────────────
# MATPLOTLIB GLOBAL STYLE
# ─────────────────────────────────────────────────────────────
BG_DARK  = "#0a0e17"
BG_PANEL = "#111827"
GRID_CLR = "#1f2937"
TICK_CLR = "#6b7280"
TXT_CLR  = "#e2e8f0"
TITLE_CLR= "#f0f4ff"

PALETTE  = ["#3b82f6", "#22c55e", "#a855f7", "#f59e0b"]   # per X


# ─────────────────────────────────────────────────────────────
# ❶  HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">📈 Implementasi <em>Regresi Linear Berganda</em></div>
  <p class="hero-sub">Multiple Linear Regression · 4 Variabel X · 1 Variabel Y</p>
  <p class="hero-note">Dataset: California Housing (Synthetic) · sklearn-style · 20,000 records</p>
  <div class="badge-row">
    <span class="badge bx1">X₁ Pendapatan_Median</span>
    <span class="badge bx2">X₂ Usia_Rumah</span>
    <span class="badge bx3">X₃ Rata_Kamar</span>
    <span class="badge bx4">X₄ Populasi</span>
    <span class="badge by">Y  Harga_Rumah</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ❷  INFORMASI DATASET
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📋 1 · Informasi Dataset</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="stat-row">
  <div class="stat-pill"><div class="sv">{len(df):,}</div><div class="sl">TOTAL RECORDS</div></div>
  <div class="stat-pill"><div class="sv">4</div><div class="sl">VARIABEL INDEPENDEN (X)</div></div>
  <div class="stat-pill"><div class="sv">1</div><div class="sl">VARIABEL DEPENDEN (Y)</div></div>
  <div class="stat-pill"><div class="sv">{df[Y_COL].mean():.3f}</div><div class="sl">RATA-RATA HARGA (×$100K)</div></div>
  <div class="stat-pill"><div class="sv">{df[Y_COL].std():.3f}</div><div class="sl">STD DEV HARGA</div></div>
</div>
""", unsafe_allow_html=True)

# Tabel keterangan variabel
st.markdown("""
<table class="var-tbl">
  <tr>
    <th>Simbol</th><th>Nama Kolom</th><th>Keterangan</th>
    <th>Satuan</th><th>Tipe</th>
  </tr>
  <tr>
    <td>X₁</td><td><code>Pendapatan_Median</code></td>
    <td>Pendapatan median penduduk per blok</td>
    <td>×$10,000 USD</td><td><span class="tx">INDEPENDEN</span></td>
  </tr>
  <tr>
    <td>X₂</td><td><code>Usia_Rumah</code></td>
    <td>Rata-rata usia bangunan rumah di blok</td>
    <td>Tahun</td><td><span class="tx">INDEPENDEN</span></td>
  </tr>
  <tr>
    <td>X₃</td><td><code>Rata_Kamar</code></td>
    <td>Rata-rata jumlah kamar per rumah tangga</td>
    <td>Unit</td><td><span class="tx">INDEPENDEN</span></td>
  </tr>
  <tr>
    <td>X₄</td><td><code>Populasi</code></td>
    <td>Total populasi di blok perumahan</td>
    <td>Jiwa</td><td><span class="tx">INDEPENDEN</span></td>
  </tr>
  <tr>
    <td>Y</td><td><code>Harga_Rumah</code></td>
    <td>Nilai median rumah di blok (target prediksi)</td>
    <td>×$100,000 USD</td><td><span class="ty">DEPENDEN</span></td>
  </tr>
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

t_prev, t_stat = st.tabs(["👁️  Preview 50 Baris Pertama", "📊  Statistik Deskriptif"])
with t_prev:
    st.dataframe(df.head(50), use_container_width=True)
with t_stat:
    st.dataframe(df.describe().round(4), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ❸  HEATMAP KORELASI
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec-hdr">🔥 2 · Heatmap Korelasi Antar Variabel</div>',
            unsafe_allow_html=True)
st.caption("Nilai mendekati +1 = korelasi positif kuat · mendekati -1 = korelasi negatif kuat · 0 = tidak berkorelasi")

corr = df.corr()

fig_heat, ax_heat = plt.subplots(figsize=(7, 5.5))
fig_heat.patch.set_facecolor(BG_DARK)
ax_heat.set_facecolor(BG_DARK)

# Label ramah baca
tick_labels = ["Pendapatan\nMedian", "Usia\nRumah", "Rata\nKamar", "Populasi", "Harga\nRumah"]

hm = sns.heatmap(
    corr,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    linewidths=0.6,
    linecolor=GRID_CLR,
    ax=ax_heat,
    annot_kws={"size": 10, "weight": "bold"},
    vmin=-1, vmax=1,
    square=True,
    xticklabels=tick_labels,
    yticklabels=tick_labels,
)

# Style colorbar
cbar = ax_heat.collections[0].colorbar
cbar.ax.tick_params(colors=TICK_CLR, labelsize=9)
cbar.outline.set_edgecolor(GRID_CLR)

ax_heat.set_title("Matriks Korelasi — California Housing Synthetic",
                  color=TITLE_CLR, fontsize=13, fontweight="bold", pad=14)
ax_heat.tick_params(colors=TXT_CLR, labelsize=9)
for sp in ax_heat.spines.values():
    sp.set_visible(False)

plt.tight_layout()

col_heat1, col_heat2 = st.columns([2.2, 1])
with col_heat1:
    st.pyplot(fig_heat)
with col_heat2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📌 Interpretasi Korelasi:**")
    for var in X_COLS:
        r_val = corr[Y_COL][var]
        if   abs(r_val) >= 0.70: strength = "🔴 Kuat"
        elif abs(r_val) >= 0.40: strength = "🟡 Sedang"
        else:                    strength = "🟢 Lemah"
        direction = "positif ↑" if r_val >= 0 else "negatif ↓"
        st.markdown(f"- **{var}** vs Y: `r = {r_val:.3f}` → {strength}, {direction}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("""
**Aturan Umum (Evans, 1996):**
- |r| 0.00–0.19 → Sangat lemah
- |r| 0.20–0.39 → Lemah
- |r| 0.40–0.59 → Sedang
- |r| 0.60–0.79 → Kuat
- |r| 0.80–1.00 → Sangat kuat
""")


# ─────────────────────────────────────────────────────────────
# ❹  SCATTER PLOT  (masing-masing X vs Y)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec-hdr">📉 3 · Scatter Plot — Setiap X terhadap Y</div>',
            unsafe_allow_html=True)
st.caption(f"Menggunakan sampel 2,000 titik acak dari total {len(df):,} data untuk keterbacaan visual. Garis putus-putus = garis tren regresi sederhana.")

sample = df.sample(2_000, random_state=42)

x_meta = {
    "Pendapatan_Median": ("Pendapatan Median  (×$10K)",  "#3b82f6"),
    "Usia_Rumah":        ("Usia Rata-rata Rumah (tahun)", "#22c55e"),
    "Rata_Kamar":        ("Rata-rata Jumlah Kamar",       "#a855f7"),
    "Populasi":          ("Populasi Blok (jiwa)",         "#f59e0b"),
}

fig_sc, axes = plt.subplots(2, 2, figsize=(13, 9))
fig_sc.patch.set_facecolor(BG_DARK)
axes = axes.flatten()

for i, (col, (xlabel, color)) in enumerate(x_meta.items()):
    ax = axes[i]
    ax.set_facecolor(BG_PANEL)

    # Scatter
    ax.scatter(
        sample[col], sample[Y_COL],
        color=color, alpha=0.45, s=14, edgecolors="none", zorder=2,
    )

    # Garis tren (regresi sederhana)
    m, b = np.polyfit(sample[col], sample[Y_COL], 1)
    x_min, x_max = sample[col].min(), sample[col].max()
    xs = np.linspace(x_min, x_max, 200)
    r_val = corr[col][Y_COL]
    ax.plot(xs, m * xs + b,
            color="white", lw=2, ls="--", alpha=0.85,
            label=f"Tren  (r = {r_val:.3f})", zorder=3)

    # Annotations
    ax.set_xlabel(xlabel, color=TICK_CLR, fontsize=10)
    ax.set_ylabel("Harga Rumah  (×$100K)", color=TICK_CLR, fontsize=10)
    ax.set_title(f"X{i+1}: {col}  vs  Y (Harga_Rumah)",
                 color=TITLE_CLR, fontsize=11, fontweight="bold")
    ax.tick_params(colors=TICK_CLR, labelsize=9)
    ax.legend(fontsize=9, facecolor="#1f2937",
              edgecolor="#374151", labelcolor="white", loc="upper left")
    ax.grid(True, color=GRID_CLR, lw=0.5, ls="--")
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)

fig_sc.suptitle(
    "Scatter Plot: Variabel Independen vs Harga Rumah",
    color=TITLE_CLR, fontsize=14, fontweight="bold", y=1.01,
)
plt.tight_layout()
st.pyplot(fig_sc)


# ─────────────────────────────────────────────────────────────
# ❺  MODEL REGRESI LINEAR BERGANDA
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec-hdr">🤖 4 · Model Regresi Linear Berganda</div>',
            unsafe_allow_html=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

b0  = model.intercept_
coefs = model.coef_    # [b1, b2, b3, b4]

# ── Persamaan regresi ────────────────────────────────────────
parts = []
for i, c in enumerate(coefs):
    sign = "+" if c >= 0 else "−"
    parts.append(f" {sign} {abs(c):.5f}·X{i+1}")

eq_str = f"Ŷ  =  {b0:+.5f}{''.join(parts)}"

st.markdown(
    f'<div class="eq-box"><b>Persamaan Regresi:</b><br>{eq_str}<br>'
    f'<span style="font-size:.78rem; color:#7ca3d1;">'
    f'Ŷ = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + β₄X₄</span></div>',
    unsafe_allow_html=True,
)

# ── Tabel koefisien ─────────────────────────────────────────
var_labels = [
    ("β₀", "Intercept",           "—",                        "Nilai Y ketika semua X = 0"),
    ("β₁", "Pendapatan_Median",   "×$10K / unit",              "Kenaikan 1 unit pendapatan → harga naik β₁×100K"),
    ("β₂", "Usia_Rumah",          "tahun / unit",              "Kenaikan 1 tahun usia → harga berubah β₂×100K"),
    ("β₃", "Rata_Kamar",          "kamar / unit",              "Kenaikan 1 kamar rata-rata → harga berubah β₃×100K"),
    ("β₄", "Populasi",            "jiwa / unit",               "Kenaikan 1 jiwa populasi → harga berubah β₄×100K"),
]
all_coefs = [b0] + list(coefs)

rows_html = ""
for (sym, name, satuan, interp), coef in zip(var_labels, all_coefs):
    css  = "cp" if coef >= 0 else "cn"
    sign = "↑ positif" if coef >= 0 else "↓ negatif"
    rows_html += (
        f"<tr>"
        f"<td>{sym}</td><td><code>{name}</code></td>"
        f"<td class='{css}'>{coef:+.6f}</td>"
        f"<td>{sign}</td><td>{satuan}</td><td>{interp}</td>"
        f"</tr>"
    )

st.markdown(f"""
<table class="coef-tbl">
  <tr>
    <th>Simbol</th><th>Variabel</th><th>Koefisien</th>
    <th>Arah</th><th>Satuan</th><th>Interpretasi</th>
  </tr>
  {rows_html}
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Variabel paling berpengaruh
sorted_coefs = sorted(zip(X_COLS, coefs), key=lambda t: abs(t[1]), reverse=True)
most_inf = sorted_coefs[0]
st.info(
    f"**Variabel paling berpengaruh:** `{most_inf[0]}` dengan |β| = **{abs(most_inf[1]):.5f}**  — "
    f"setiap kenaikan 1 unit, harga rumah berubah sebesar **${abs(most_inf[1])*100_000:,.0f}** "
    f"({'naik' if most_inf[1] > 0 else 'turun'})."
)


# ─────────────────────────────────────────────────────────────
# ❻  EVALUASI MODEL
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec-hdr">📊 5 · Evaluasi Model</div>', unsafe_allow_html=True)
st.caption(f"Data uji: {len(y_test):,} records (20% dari total) · Data latih: {len(y_train):,} records (80%)")

r2     = r2_score(y_test, y_pred)
n, p   = len(y_test), len(X_COLS)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
mae    = mean_absolute_error(y_test, y_pred)
mse    = mean_squared_error(y_test, y_pred)

# Metric cards
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
metrics = [
    (mc1, "#3b82f6", f"{r2:.4f}",     "R² SCORE",       f"Model menjelaskan {r2*100:.1f}% variansi Y"),
    (mc2, "#22c55e", f"{adj_r2:.4f}", "ADJUSTED R²",    "R² dikoreksi jumlah prediktor"),
    (mc3, "#f59e0b", f"{rmse:.4f}",   "RMSE",           f"≈ ${rmse*100_000:,.0f} rata-rata error"),
    (mc4, "#a855f7", f"{mae:.4f}",    "MAE",            f"≈ ${mae*100_000:,.0f} deviasi absolut"),
    (mc5, "#f87171", f"{mse:.4f}",    "MSE",            "Mean Squared Error"),
]
for col, border, val, label, desc in metrics:
    col.markdown(
        f'<div class="mc" style="border-left-color:{border};">'
        f'<div class="mv">{val}</div>'
        f'<div class="ml">{label}</div>'
        f'<div class="md">{desc}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Plot Evaluasi 3-panel ────────────────────────────────────
fig_eval, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig_eval.patch.set_facecolor(BG_DARK)

residuals = y_test - y_pred

# — Plot 1: Aktual vs Prediksi —
ax1.set_facecolor(BG_PANEL)
ax1.scatter(y_test, y_pred, color="#3b82f6", alpha=0.30, s=10, edgecolors="none", zorder=2)
lim_lo = min(float(y_test.min()), float(y_pred.min()))
lim_hi = max(float(y_test.max()), float(y_pred.max()))
ax1.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
         color="#ef4444", lw=2, ls="--", label="Ideal (y = ŷ)", zorder=3)
ax1.set_xlabel("Nilai Aktual", color=TICK_CLR, fontsize=10)
ax1.set_ylabel("Nilai Prediksi", color=TICK_CLR, fontsize=10)
ax1.set_title("Aktual  vs  Prediksi", color=TITLE_CLR, fontsize=12, fontweight="bold")
ax1.tick_params(colors=TICK_CLR, labelsize=9)
ax1.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white", fontsize=9)
ax1.grid(True, color=GRID_CLR, lw=0.5, ls="--")
ax1.text(0.04, 0.94, f"R² = {r2:.4f}", transform=ax1.transAxes,
         color="#93c5fd", fontsize=10, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#0c1f3a", edgecolor="#1d4ed8"))
for sp in ax1.spines.values(): sp.set_edgecolor(GRID_CLR)

# — Plot 2: Residual vs Prediksi —
ax2.set_facecolor(BG_PANEL)
ax2.scatter(y_pred, residuals, color="#f59e0b", alpha=0.30, s=10, edgecolors="none", zorder=2)
ax2.axhline(0, color="#ef4444", lw=2, ls="--", label="Residual = 0", zorder=3)
ax2.set_xlabel("Nilai Prediksi (Ŷ)", color=TICK_CLR, fontsize=10)
ax2.set_ylabel("Residual  (Y − Ŷ)", color=TICK_CLR, fontsize=10)
ax2.set_title("Plot Residual", color=TITLE_CLR, fontsize=12, fontweight="bold")
ax2.tick_params(colors=TICK_CLR, labelsize=9)
ax2.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white", fontsize=9)
ax2.grid(True, color=GRID_CLR, lw=0.5, ls="--")
for sp in ax2.spines.values(): sp.set_edgecolor(GRID_CLR)

# — Plot 3: Distribusi Residual —
ax3.set_facecolor(BG_PANEL)
ax3.hist(residuals, bins=60, color="#a855f7", alpha=0.80, edgecolor=BG_DARK)
ax3.axvline(0, color="#ef4444", lw=2, ls="--", label="Residual = 0")
ax3.set_xlabel("Residual", color=TICK_CLR, fontsize=10)
ax3.set_ylabel("Frekuensi", color=TICK_CLR, fontsize=10)
ax3.set_title("Distribusi Residual", color=TITLE_CLR, fontsize=12, fontweight="bold")
ax3.tick_params(colors=TICK_CLR, labelsize=9)
ax3.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="white", fontsize=9)
ax3.grid(True, color=GRID_CLR, lw=0.5, ls="--", axis="y")
ax3.text(0.04, 0.94, f"μ = {residuals.mean():.4f}\nσ = {residuals.std():.4f}",
         transform=ax3.transAxes, color="#c4b5fd", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1b0d2e", edgecolor="#7c3aed"))
for sp in ax3.spines.values(): sp.set_edgecolor(GRID_CLR)

plt.tight_layout(pad=2.5)
st.pyplot(fig_eval)

# Penjelasan plot
col_ex1, col_ex2, col_ex3 = st.columns(3)
col_ex1.caption("💡 Titik-titik mendekati garis merah = prediksi mendekati nilai asli. Penyebaran simetris = model tidak bias.")
col_ex2.caption("💡 Residual tersebar acak di sekitar nol (tidak ada pola) = asumsi homoskedastisitas terpenuhi.")
col_ex3.caption("💡 Distribusi residual yang mendekati normal (lonceng simetris) menandakan model sudah fit dengan baik.")


# ─────────────────────────────────────────────────────────────
# ❼  RINGKASAN & KESIMPULAN
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec-hdr">📝 6 · Kesimpulan & Interpretasi Model</div>',
            unsafe_allow_html=True)

st.markdown(f"""
| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **R² Score** | **{r2:.4f}** | Model menjelaskan **{r2*100:.1f}%** variasi harga rumah |
| **Adjusted R²** | **{adj_r2:.4f}** | Konsisten dengan R² → variabel prediktor relevan |
| **RMSE** | **{rmse:.4f}** | Rata-rata error prediksi ≈ **${rmse*100_000:,.0f}** per rumah |
| **MAE** | **{mae:.4f}** | Rata-rata deviasi absolut ≈ **${mae*100_000:,.0f}** per rumah |
| **MSE** | **{mse:.4f}** | Mean Squared Error (sensitif terhadap outlier) |
""")

# Peringkat pengaruh variabel
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### 🏆 Peringkat Pengaruh Variabel (berdasarkan |koefisien|)")

rank_rows = ""
for rank, (var, coef) in enumerate(sorted_coefs, start=1):
    bar = "█" * min(int(abs(coef) / max(abs(c) for _, c in sorted_coefs) * 20), 20)
    css = "cp" if coef >= 0 else "cn"
    rank_rows += (
        f"<tr>"
        f"<td>#{rank}</td><td><code>{var}</code></td>"
        f"<td class='{css}'>{coef:+.6f}</td>"
        f"<td><span style='color:#3b82f6; font-family:monospace; letter-spacing:-1px;'>{bar}</span></td>"
        f"</tr>"
    )

st.markdown(f"""
<table class="coef-tbl">
  <tr><th>Peringkat</th><th>Variabel</th><th>Koefisien</th><th>Pengaruh Relatif</th></tr>
  {rank_rows}
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Interpretasi lengkap
st.markdown(f"""
<div class="interp-box">
  <b>📌 Persamaan Akhir:</b><br>
  <code style="color:#93c5fd;">{eq_str}</code><br><br>
  <b>🔍 Temuan Utama:</b><br>
  1. <b>Pendapatan_Median (X₁)</b> adalah prediktor terkuat dengan β₁ = <b>{coefs[0]:+.5f}</b> —
     setiap kenaikan $10,000 pendapatan median, harga rumah naik rata-rata ${coefs[0]*100_000:,.0f}.<br>
  2. <b>Rata_Kamar (X₃)</b> memiliki pengaruh positif β₃ = <b>{coefs[2]:+.5f}</b> —
     semakin banyak kamar rata-rata, harga rumah cenderung lebih tinggi.<br>
  3. <b>Usia_Rumah (X₂)</b> memiliki pengaruh positif kecil β₂ = <b>{coefs[1]:+.5f}</b>.<br>
  4. <b>Populasi (X₄)</b> memiliki pengaruh negatif sangat kecil β₄ = <b>{coefs[3]:+.7f}</b> —
     kepadatan populasi sedikit menekan harga rumah.<br><br>
  <b>✅ Kualitas Model:</b> R² = <b>{r2:.4f}</b> menunjukkan model <b>{'sangat baik' if r2 >= 0.80 else 'baik' if r2 >= 0.60 else 'cukup'}</b>
  dalam menjelaskan harga rumah berdasarkan 4 variabel yang dipilih.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "📚 Referensi: Evans, J.D. (1996). Straightforward statistics for the behavioral sciences. "
    "Dataset: California Housing Synthetic (generated with numpy, seed=42, n=20,000)."
)
