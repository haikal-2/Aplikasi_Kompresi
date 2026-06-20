"""
==========================================================
Implementasi Regresi Linear Berganda
Dataset  : Video Game Sales  (vgsales.csv — Kaggle)
Variabel : X1 NA_Sales   X2 EU_Sales   X3 JP_Sales
           Y  Global_Sales
==========================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "vgsales.csv"
SAMPLE_N  = 500
SEED      = 42
X_COLS    = ["NA_Sales", "EU_Sales", "JP_Sales"]
Y_COL     = "Global_Sales"


# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regresi Linear Berganda",
    layout="wide",
    page_icon=None,
)

# ──────────────────────────────────────────────────────────────
# CSS  —  Hitam-Putih / Mathematical Monograph Style
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, .stApp {
    background-color: #0c0c0c !important;
    color: #e8e8e8;
    font-family: 'Georgia', 'Times New Roman', serif;
  }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem 4rem; max-width: 1360px; }

  .page-header {
    border-top: 3px solid #ffffff;
    border-bottom: 1px solid #2a2a2a;
    padding: 2.2rem 0 1.8rem;
    margin-bottom: 2.5rem;
    text-align: center;
  }
  .page-header h1 {
    font-size: 2rem; font-weight: 700; color: #ffffff;
    letter-spacing: 1px; margin: 0 0 .5rem;
    font-family: 'Georgia', serif;
    text-transform: uppercase;
  }
  .page-header p { color: #666666; font-size: .9rem; margin: 0; font-family: 'Courier New', monospace; }
  .page-header .sub { color: #444444; font-size: .78rem; margin-top: .4rem; letter-spacing: .5px; }

  .tag-row { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 1.2rem; }
  .tag {
    font-family: 'Courier New', monospace;
    font-size: .72rem; font-weight: 700; letter-spacing: .8px;
    padding: 3px 12px; border: 1px solid #333333; color: #aaaaaa;
    text-transform: uppercase;
  }
  .tag-y { border-color: #888888; color: #ffffff; }

  .sec-hdr {
    font-family: 'Georgia', serif;
    font-size: 1rem; font-weight: 700;
    color: #ffffff; letter-spacing: .5px;
    text-transform: uppercase;
    padding-bottom: .5rem;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 1.4rem;
  }
  .sec-num { font-family: 'Courier New', monospace; color: #555555; margin-right: .6rem; font-size: .85rem; }

  .stat-strip { display: flex; gap: 10px; margin-bottom: 1.5rem; }
  .stat-box {
    background: #111111; border: 1px solid #222222;
    padding: .8rem 1rem; flex: 1; text-align: center;
  }
  .sv { font-size: 1.5rem; font-weight: 700; color: #ffffff; font-family: 'Courier New', monospace; }
  .sl { font-size: .66rem; color: #555555; letter-spacing: .8px; margin-top: 3px;
        font-family: 'Courier New', monospace; text-transform: uppercase; }

  .pipe-row { display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .pipe-step {
    background: #111111; border: 1px solid #222222;
    padding: .6rem 1rem; text-align: center; flex: 1; min-width: 140px;
  }
  .pipe-step .pv { font-size: 1.2rem; font-weight: 700; color: #ffffff; font-family: 'Courier New', monospace; }
  .pipe-step .pl { font-size: .65rem; color: #555555; margin-top: 2px; font-family: 'Courier New', monospace;
                    text-transform: uppercase; letter-spacing: .5px; }
  .pipe-arrow { color: #444444; font-family: 'Courier New', monospace; font-size: 1rem; }

  .var-tbl { width: 100%; border-collapse: collapse; font-size: .83rem;
             margin-bottom: 1rem; font-family: 'Courier New', monospace; }
  .var-tbl th {
    background: #161616; color: #aaaaaa; padding: 8px 14px;
    font-weight: 700; text-align: left;
    border-bottom: 2px solid #2a2a2a; letter-spacing: .5px;
  }
  .var-tbl td { background: #111111; color: #cccccc; padding: 7px 14px; border-bottom: 1px solid #1a1a1a; }
  .var-tbl tr:hover td { background: #161616; }
  .tx { border: 1px solid #333333; color: #999999; padding: 2px 8px; font-size: .68rem; font-weight: 700; letter-spacing: .5px; }
  .ty { border: 1px solid #777777; color: #ffffff; padding: 2px 8px; font-size: .68rem; font-weight: 700; letter-spacing: .5px; }

  .coef-tbl { width: 100%; border-collapse: collapse; font-size: .83rem; font-family: 'Courier New', monospace; }
  .coef-tbl th {
    background: #111111; color: #aaaaaa; padding: 9px 14px;
    font-weight: 700; text-align: left; border-bottom: 2px solid #2a2a2a; letter-spacing: .5px;
  }
  .coef-tbl td { padding: 8px 14px; border-bottom: 1px solid #1a1a1a; color: #cccccc; }
  .coef-tbl tr:hover td { background: #111111; }
  .cp { color: #ffffff !important; font-weight: 700; }
  .cn { color: #888888 !important; font-weight: 700; }

  .mc {
    background: #111111; border: 1px solid #222222;
    border-left: 3px solid #ffffff;
    padding: 1rem 1.2rem; text-align: center;
  }
  .mc .mv { font-size: 1.5rem; font-weight: 700; color: #ffffff; font-family: 'Courier New', monospace; }
  .mc .ml { font-size: .66rem; color: #555555; letter-spacing: .8px; margin-top: 2px;
             font-family: 'Courier New', monospace; text-transform: uppercase; }
  .mc .md { font-size: .75rem; color: #777777; margin-top: 5px; font-family: 'Georgia', serif; }
  .mc-dim { border-left-color: #444444; }
  .mc-dim .mv { color: #aaaaaa; }

  .interp-box {
    background: #0f0f0f; border: 1px solid #222222;
    border-left: 3px solid #ffffff;
    padding: 1.2rem 1.6rem; font-size: .86rem;
    line-height: 1.9; color: #bbbbbb;
    font-family: 'Georgia', serif;
  }
  .interp-box b { color: #ffffff; }
  .interp-box code { font-family: 'Courier New', monospace; background: #1a1a1a; padding: 1px 6px; color: #dddddd; }

  .eq-label {
    font-family: 'Courier New', monospace;
    font-size: .72rem; color: #555555;
    letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: .5rem;
  }

  .stTabs [data-baseweb="tab-list"] { background: #111111; padding: 3px; border: 1px solid #222222; gap: 1px; }
  .stTabs [data-baseweb="tab"] {
    background: transparent; color: #555555; font-weight: 600; font-size: .82rem;
    font-family: 'Courier New', monospace; letter-spacing: .3px;
  }
  .stTabs [aria-selected="true"] { background: #1e1e1e !important; color: #ffffff !important; }

  hr { border-color: #1e1e1e !important; margin: 2.5rem 0 !important; }
  .stDataFrame { font-family: 'Courier New', monospace !important; }
  div[data-testid="metric-container"] { background: #111111; border: 1px solid #222222; padding: 1rem; }
  .stInfo, .stError { background: #111111 !important; border: 1px solid #333333 !important; }
  .katex { color: #e8e8e8 !important; }
  .katex-display { margin: .5rem 0 !important; }
  .stCaption p { color: #444444 !important; font-family: 'Courier New', monospace; font-size: .74rem !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# MATPLOTLIB — Grayscale Mathematical Style
# ──────────────────────────────────────────────────────────────
BG    = "#0c0c0c"
PANEL = "#111111"
GRID  = "#1a1a1a"
TICK  = "#555555"
TXT   = "#cccccc"
WHITE = "#f0f0f0"
GRAY1 = "#888888"
GRAY2 = "#555555"

BW_CMAP = LinearSegmentedColormap.from_list("bw_div", ["#1a1a1a", "#555555", "#f0f0f0"])


# ──────────────────────────────────────────────────────────────
# DATA PIPELINE
# ──────────────────────────────────────────────────────────────
def load_raw_data(path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari berkas CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan dataset:
      - Membuang baris dengan nilai Year yang kosong.
      - Membuang baris dengan Publisher kosong atau bernilai 'Unknown'.
    """
    df = df.dropna(subset=["Year", "Publisher"])
    df = df[df["Publisher"] != "Unknown"]
    return df.reset_index(drop=True)


def sample_data(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Mengambil sampel acak sejumlah n baris dengan random_state tetap."""
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


@st.cache_data
def prepare_dataset(path: str, n: int, seed: int):
    raw     = load_raw_data(path)
    cleaned = clean_data(raw)
    sample  = sample_data(cleaned, n, seed)
    return raw, cleaned, sample


try:
    raw_df, cleaned_df, df = prepare_dataset(DATA_PATH, SAMPLE_N, SEED)
except FileNotFoundError:
    st.error(
        f"Berkas '{DATA_PATH}' tidak ditemukan. "
        f"Pastikan berkas tersebut berada pada direktori yang sama dengan skrip ini."
    )
    st.stop()

X, y = df[X_COLS], df[Y_COL]


# ──────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>Regresi Linear Berganda</h1>
  <p>Multiple Linear Regression  —  Video Game Sales Dataset</p>
  <p class="sub">Sumber Data : vgsales.csv (Kaggle)  ·  Rentang Tahun 1980 - 2020</p>
  <div class="tag-row">
    <span class="tag">X1  NA_Sales</span>
    <span class="tag">X2  EU_Sales</span>
    <span class="tag">X3  JP_Sales</span>
    <span class="tag tag-y">Y   Global_Sales</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  I.  INFORMASI DATASET
# ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="sec-hdr"><span class="sec-num">I.</span>Informasi Dataset</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="eq-label">Tahapan Pemrosesan Data</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="pipe-row">
  <div class="pipe-step"><div class="pv">{len(raw_df):,}</div><div class="pl">Baris Mentah</div></div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="pipe-step"><div class="pv">{len(cleaned_df):,}</div><div class="pl">Setelah Pembersihan</div></div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="pipe-step"><div class="pv">{len(df):,}</div><div class="pl">Sampel Digunakan</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stat-strip">
  <div class="stat-box"><div class="sv">{len(df):,}</div><div class="sl">Total Records (Sampel)</div></div>
  <div class="stat-box"><div class="sv">3</div><div class="sl">Variabel Independen</div></div>
  <div class="stat-box"><div class="sv">1</div><div class="sl">Variabel Dependen</div></div>
  <div class="stat-box"><div class="sv">{df[Y_COL].mean():.3f}</div><div class="sl">Mean Y (Juta Unit)</div></div>
  <div class="stat-box"><div class="sv">{df[Y_COL].std():.3f}</div><div class="sl">Std Dev Y</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<table class="var-tbl">
  <tr><th>Simbol</th><th>Kolom</th><th>Keterangan</th><th>Satuan</th><th>Tipe</th></tr>
  <tr>
    <td>X1</td><td>NA_Sales</td>
    <td>Total penjualan game di kawasan Amerika Utara</td>
    <td>Juta unit</td><td><span class="tx">Independen</span></td>
  </tr>
  <tr>
    <td>X2</td><td>EU_Sales</td>
    <td>Total penjualan game di kawasan Eropa</td>
    <td>Juta unit</td><td><span class="tx">Independen</span></td>
  </tr>
  <tr>
    <td>X3</td><td>JP_Sales</td>
    <td>Total penjualan game di kawasan Jepang</td>
    <td>Juta unit</td><td><span class="tx">Independen</span></td>
  </tr>
  <tr>
    <td>Y</td><td>Global_Sales</td>
    <td>Total penjualan game secara global (target prediksi)</td>
    <td>Juta unit</td><td><span class="ty">Dependen</span></td>
  </tr>
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

t1, t2 = st.tabs(["Preview Data Sampel", "Statistik Deskriptif"])
with t1:
    preview_cols = ["Name", "Platform", "Year", "Genre"] + X_COLS + [Y_COL]
    st.dataframe(df[preview_cols], use_container_width=True, height=380)
with t2:
    st.dataframe(df[X_COLS + [Y_COL]].describe().round(4), use_container_width=True)


# ──────────────────────────────────────────────────────────────
#  II.  HEATMAP KORELASI
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="sec-hdr"><span class="sec-num">II.</span>Heatmap Korelasi Antar Variabel</div>',
    unsafe_allow_html=True,
)
st.caption("Nilai r mendekati +1 : korelasi positif kuat.  Mendekati -1 : korelasi negatif kuat.  Mendekati 0 : tidak berkorelasi.")

corr = df[X_COLS + [Y_COL]].corr()
tick_lbl = ["NA Sales", "EU Sales", "JP Sales", "Global Sales"]

fig_h, ax_h = plt.subplots(figsize=(6.6, 5.4))
fig_h.patch.set_facecolor(BG)
ax_h.set_facecolor(BG)

sns.heatmap(
    corr,
    annot=True, fmt=".3f",
    cmap=BW_CMAP,
    linewidths=0.5, linecolor="#1e1e1e",
    ax=ax_h,
    annot_kws={"size": 11, "weight": "bold", "color": "#0c0c0c"},
    vmin=-1, vmax=1, square=True,
    xticklabels=tick_lbl, yticklabels=tick_lbl,
)
cbar = ax_h.collections[0].colorbar
cbar.ax.tick_params(colors=TICK, labelsize=9)
cbar.outline.set_edgecolor(GRID)
ax_h.set_title("Matriks Korelasi — Video Game Sales",
               color=WHITE, fontsize=12, fontweight="bold", fontfamily="Georgia", pad=14)
ax_h.tick_params(colors=TXT, labelsize=9)
for sp in ax_h.spines.values():
    sp.set_visible(False)
plt.tight_layout()

col_h1, col_h2 = st.columns([2.1, 1])
with col_h1:
    st.pyplot(fig_h)
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Nilai korelasi (r) dengan Y :**")
    for var in X_COLS:
        r_val = corr[Y_COL][var]
        if   abs(r_val) >= 0.70: kuat = "Kuat"
        elif abs(r_val) >= 0.40: kuat = "Sedang"
        else:                    kuat = "Lemah"
        arah = "positif" if r_val >= 0 else "negatif"
        st.markdown(
            f"<span style='font-family:Courier New;font-size:.82rem;color:#888'>{var}</span>"
            f"<br><span style='font-family:Courier New;font-size:.9rem;color:#fff;font-weight:700;'>"
            f"r = {r_val:.4f}</span>"
            f"<span style='font-size:.75rem;color:#555;'> — {kuat}, {arah}</span><br><br>",
            unsafe_allow_html=True,
        )
    st.markdown("""
<div style="border:1px solid #222; padding:.8rem 1rem; font-family:'Courier New',monospace; font-size:.75rem; color:#666; line-height:1.9;">
<b style="color:#aaa;">Evans (1996)</b><br>
|r| 0.00 - 0.19 : Sangat lemah<br>
|r| 0.20 - 0.39 : Lemah<br>
|r| 0.40 - 0.59 : Sedang<br>
|r| 0.60 - 0.79 : Kuat<br>
|r| 0.80 - 1.00 : Sangat kuat
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  III.  SCATTER PLOT
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="sec-hdr"><span class="sec-num">III.</span>Scatter Plot  —  Setiap X terhadap Y</div>',
    unsafe_allow_html=True,
)
st.caption(f"Seluruh {len(df):,} titik pada sampel ditampilkan.  Garis putus-putus = garis tren regresi sederhana.")

x_meta = {
    "NA_Sales": "NA Sales  (Juta Unit)",
    "EU_Sales": "EU Sales  (Juta Unit)",
    "JP_Sales": "JP Sales  (Juta Unit)",
}

fig_sc, axes = plt.subplots(1, 3, figsize=(15, 5))
fig_sc.patch.set_facecolor(BG)

for i, (col, xlabel) in enumerate(x_meta.items()):
    ax = axes[i]
    ax.set_facecolor(PANEL)

    ax.scatter(df[col], df[Y_COL],
               color=GRAY1, alpha=0.45, s=18, edgecolors="none", zorder=2)

    m, b = np.polyfit(df[col], df[Y_COL], 1)
    xs   = np.linspace(df[col].min(), df[col].max(), 200)
    r_val = corr[col][Y_COL]
    ax.plot(xs, m * xs + b,
            color=WHITE, lw=1.8, ls="--", alpha=0.9,
            label=f"r = {r_val:.3f}", zorder=3)

    ax.set_xlabel(xlabel, color=TICK, fontsize=9, fontfamily="Courier New")
    ax.set_ylabel("Global Sales  (Juta Unit)", color=TICK, fontsize=9, fontfamily="Courier New")
    ax.set_title(f"X{i+1}: {col}  vs  Y",
                 color=WHITE, fontsize=11, fontweight="bold", fontfamily="Georgia")
    ax.tick_params(colors=TICK, labelsize=8)
    ax.legend(fontsize=9, facecolor="#161616", edgecolor="#2a2a2a", labelcolor=WHITE,
              prop={"family": "Courier New"})
    ax.grid(True, color=GRID, lw=0.5, ls="--")
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

fig_sc.suptitle("Scatter Plot : Penjualan Regional  vs  Penjualan Global",
                color=WHITE, fontsize=13, fontweight="bold", fontfamily="Georgia", y=1.03)
plt.tight_layout()
st.pyplot(fig_sc)


# ──────────────────────────────────────────────────────────────
#  IV.  MODEL REGRESI
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="sec-hdr"><span class="sec-num">IV.</span>Model Regresi Linear Berganda</div>',
    unsafe_allow_html=True,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
model  = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

b0    = model.intercept_
coefs = model.coef_

st.markdown('<div class="eq-label">Bentuk Umum Persamaan Regresi</div>', unsafe_allow_html=True)
st.latex(r"\hat{Y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \varepsilon")

st.markdown("<br>", unsafe_allow_html=True)

sign = [("+" if c >= 0 else "-") for c in coefs]
cabs = [abs(v) for v in coefs]
st.markdown('<div class="eq-label">Persamaan dengan Nilai Koefisien</div>', unsafe_allow_html=True)
st.latex(
    rf"\hat{{Y}} = {b0:.5f}"
    rf"\ {sign[0]}\ {cabs[0]:.5f}\,X_1"
    rf"\ {sign[1]}\ {cabs[1]:.5f}\,X_2"
    rf"\ {sign[2]}\ {cabs[2]:.5f}\,X_3"
)

st.markdown("<br>", unsafe_allow_html=True)

var_meta = [
    ("b0", "Intercept (Konstan)", "—",            "Nilai Y ketika seluruh X = 0"),
    ("b1", "NA_Sales",            "unit / unit",  "Naik 1 juta unit NA_Sales -> Global_Sales naik b1 juta unit"),
    ("b2", "EU_Sales",            "unit / unit",  "Naik 1 juta unit EU_Sales -> Global_Sales naik b2 juta unit"),
    ("b3", "JP_Sales",            "unit / unit",  "Naik 1 juta unit JP_Sales -> Global_Sales naik b3 juta unit"),
]
all_coefs = [b0] + list(coefs)
rows_html = ""
for (sym, name, sat, interp), coef in zip(var_meta, all_coefs):
    css = "cp" if coef >= 0 else "cn"
    rows_html += (
        f"<tr><td>{sym}</td><td>{name}</td>"
        f"<td class='{css}'>{coef:+.6f}</td>"
        f"<td>{'positif' if coef >= 0 else 'negatif'}</td>"
        f"<td>{sat}</td><td>{interp}</td></tr>"
    )

st.markdown(f"""
<table class="coef-tbl">
  <tr><th>Simbol</th><th>Variabel</th><th>Koefisien</th><th>Arah</th><th>Satuan</th><th>Interpretasi</th></tr>
  {rows_html}
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

sorted_coefs = sorted(zip(X_COLS, coefs), key=lambda t: abs(t[1]), reverse=True)
most_inf     = sorted_coefs[0]
st.markdown(
    f'<div class="interp-box">'
    f'<b>Variabel paling berpengaruh :</b> <code>{most_inf[0]}</code> '
    f'dengan |beta| = <b>{abs(most_inf[1]):.5f}</b> — '
    f'setiap kenaikan 1 juta unit penjualan di wilayah tersebut, '
    f'penjualan global <b>{"naik" if most_inf[1] > 0 else "turun"}</b> '
    f'sekitar <b>{abs(most_inf[1]):.3f} juta unit</b>.'
    f'</div>',
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
#  V.  EVALUASI MODEL
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="sec-hdr"><span class="sec-num">V.</span>Evaluasi Model</div>',
    unsafe_allow_html=True,
)
st.caption(
    f"Data latih : {len(y_train):,} records (80%)   |   "
    f"Data uji : {len(y_test):,} records (20%)"
)

r2     = r2_score(y_test, y_pred)
n_t, p = len(y_test), len(X_COLS)
adj_r2 = 1 - (1 - r2) * (n_t - 1) / (n_t - p - 1)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
mae    = mean_absolute_error(y_test, y_pred)
mse    = mean_squared_error(y_test, y_pred)

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
metrics = [
    (mc1, False, f"{r2:.4f}",     "R-squared",   f"Model menjelaskan {r2*100:.2f}% variasi Y"),
    (mc2, False, f"{adj_r2:.4f}", "Adjusted R2", "R2 terkoreksi jumlah prediktor"),
    (mc3, True,  f"{rmse:.4f}",   "RMSE",        f"Rata-rata error : {rmse:.3f} juta unit"),
    (mc4, True,  f"{mae:.4f}",    "MAE",         f"Deviasi absolut : {mae:.3f} juta unit"),
    (mc5, True,  f"{mse:.4f}",    "MSE",         "Sensitif terhadap outlier"),
]
for col, dim, val, lbl, desc in metrics:
    css_extra = "mc-dim" if dim else ""
    col.markdown(
        f'<div class="mc {css_extra}">'
        f'<div class="mv">{val}</div>'
        f'<div class="ml">{lbl}</div>'
        f'<div class="md">{desc}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

residuals = y_test - y_pred
fig_ev, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig_ev.patch.set_facecolor(BG)

ax1.set_facecolor(PANEL)
ax1.scatter(y_test, y_pred, color=GRAY1, alpha=0.35, s=14, edgecolors="none", zorder=2)
lo = min(float(y_test.min()), float(y_pred.min()))
hi = max(float(y_test.max()), float(y_pred.max()))
ax1.plot([lo, hi], [lo, hi], color=WHITE, lw=1.8, ls="--", label="y = y_hat", zorder=3)
ax1.set_xlabel("Nilai Aktual", color=TICK, fontsize=10, fontfamily="Courier New")
ax1.set_ylabel("Nilai Prediksi", color=TICK, fontsize=10, fontfamily="Courier New")
ax1.set_title("Aktual  vs  Prediksi", color=WHITE, fontsize=12, fontweight="bold", fontfamily="Georgia")
ax1.tick_params(colors=TICK, labelsize=9)
ax1.legend(facecolor="#161616", edgecolor="#2a2a2a", labelcolor=WHITE, fontsize=9, prop={"family": "Courier New"})
ax1.grid(True, color=GRID, lw=0.5, ls="--")
ax1.text(0.04, 0.94, f"R2 = {r2:.4f}", transform=ax1.transAxes,
         color=WHITE, fontsize=10, fontweight="bold", fontfamily="Courier New",
         bbox=dict(boxstyle="square,pad=0.3", facecolor="#161616", edgecolor="#2a2a2a"))
for sp in ax1.spines.values(): sp.set_edgecolor(GRID)

ax2.set_facecolor(PANEL)
ax2.scatter(y_pred, residuals, color=GRAY1, alpha=0.35, s=14, edgecolors="none", zorder=2)
ax2.axhline(0, color=WHITE, lw=1.8, ls="--", label="Residual = 0", zorder=3)
ax2.set_xlabel("Nilai Prediksi", color=TICK, fontsize=10, fontfamily="Courier New")
ax2.set_ylabel("Residual  (Y - Y_hat)", color=TICK, fontsize=10, fontfamily="Courier New")
ax2.set_title("Plot Residual", color=WHITE, fontsize=12, fontweight="bold", fontfamily="Georgia")
ax2.tick_params(colors=TICK, labelsize=9)
ax2.legend(facecolor="#161616", edgecolor="#2a2a2a", labelcolor=WHITE, fontsize=9, prop={"family": "Courier New"})
ax2.grid(True, color=GRID, lw=0.5, ls="--")
for sp in ax2.spines.values(): sp.set_edgecolor(GRID)

ax3.set_facecolor(PANEL)
ax3.hist(residuals, bins=40, color=GRAY2, alpha=0.90, edgecolor=PANEL)
ax3.axvline(0, color=WHITE, lw=1.8, ls="--", label="Residual = 0")
ax3.set_xlabel("Residual", color=TICK, fontsize=10, fontfamily="Courier New")
ax3.set_ylabel("Frekuensi", color=TICK, fontsize=10, fontfamily="Courier New")
ax3.set_title("Distribusi Residual", color=WHITE, fontsize=12, fontweight="bold", fontfamily="Georgia")
ax3.tick_params(colors=TICK, labelsize=9)
ax3.legend(facecolor="#161616", edgecolor="#2a2a2a", labelcolor=WHITE, fontsize=9, prop={"family": "Courier New"})
ax3.grid(True, color=GRID, lw=0.5, ls="--", axis="y")
ax3.text(0.04, 0.94, f"mu = {residuals.mean():.4f}\nsigma = {residuals.std():.4f}",
         transform=ax3.transAxes, color=TXT, fontsize=9, fontfamily="Courier New",
         bbox=dict(boxstyle="square,pad=0.3", facecolor="#161616", edgecolor="#2a2a2a"))
for sp in ax3.spines.values(): sp.set_edgecolor(GRID)

plt.tight_layout(pad=2.5)
st.pyplot(fig_ev)

col_c1, col_c2, col_c3 = st.columns(3)
col_c1.caption("Titik mendekati garis diagonal = prediksi mendekati nilai aktual.")
col_c2.caption("Residual tersebar acak di sekitar nol = asumsi homoskedastisitas terpenuhi.")
col_c3.caption("Distribusi residual yang sempit menunjukkan error prediksi relatif kecil dan konsisten.")


# ──────────────────────────────────────────────────────────────
#  VI.  KESIMPULAN
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="sec-hdr"><span class="sec-num">VI.</span>Kesimpulan dan Interpretasi</div>',
    unsafe_allow_html=True,
)

st.markdown(f"""
| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| R2 Score | **{r2:.4f}** | Model menjelaskan **{r2*100:.2f}%** variasi penjualan global |
| Adjusted R2 | **{adj_r2:.4f}** | Konsisten dengan R2 — seluruh variabel relevan |
| RMSE | **{rmse:.4f}** | Rata-rata error prediksi sekitar **{rmse:.3f} juta unit** |
| MAE | **{mae:.4f}** | Rata-rata deviasi absolut sekitar **{mae:.3f} juta unit** |
| MSE | **{mse:.4f}** | Mean Squared Error (lebih sensitif terhadap outlier) |
""")

st.markdown("<br>", unsafe_allow_html=True)

rank_rows = ""
max_abs = max(abs(c) for _, c in sorted_coefs)
for rank, (var, coef) in enumerate(sorted_coefs, start=1):
    bar_len = max(1, int(abs(coef) / max_abs * 18))
    bar     = "\u2588" * bar_len + "\u2591" * (18 - bar_len)
    css     = "cp" if coef >= 0 else "cn"
    rank_rows += (
        f"<tr><td>#{rank}</td><td>{var}</td>"
        f"<td class='{css}'>{coef:+.6f}</td>"
        f"<td style='font-family:Courier New;letter-spacing:-1px;color:#444'>{bar}</td></tr>"
    )

st.markdown("**Peringkat Pengaruh Variabel berdasarkan nilai absolut koefisien :**")
st.markdown(f"""
<table class="coef-tbl">
  <tr><th>Peringkat</th><th>Variabel</th><th>Koefisien</th><th>Pengaruh Relatif</th></tr>
  {rank_rows}
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

kualitas = "sangat baik" if r2 >= 0.80 else "baik" if r2 >= 0.60 else "cukup"
st.markdown(f"""
<div class="interp-box">
  <b>Temuan Utama :</b><br>
  1. Ketiga variabel (<code>NA_Sales</code>, <code>EU_Sales</code>, <code>JP_Sales</code>) secara bersama-sama
     memiliki hubungan linear yang sangat kuat terhadap <code>Global_Sales</code>,
     karena ketiganya merupakan komponen penyusun penjualan global secara langsung.<br>
  2. <code>EU_Sales (X2)</code> memberikan kontribusi terbesar dengan b2 = <b>{coefs[1]:+.5f}</b>,
     diikuti oleh <code>NA_Sales (X1)</code> dengan b1 = <b>{coefs[0]:+.5f}</b>.<br>
  3. <code>JP_Sales (X3)</code> tetap berkontribusi positif dengan b3 = <b>{coefs[2]:+.5f}</b>,
     meskipun korelasinya terhadap Y relatif lebih lemah dibanding dua variabel lainnya.<br>
  4. Residual yang tersisa terutama berasal dari <code>Other_Sales</code> —
     komponen penjualan regional lain yang tidak disertakan sebagai variabel prediktor pada model ini.<br><br>
  <b>Kualitas Model :</b> R2 = <b>{r2:.4f}</b> — model dinilai <b>{kualitas}</b>
  dalam menjelaskan variasi penjualan global berdasarkan ketiga variabel penjualan regional yang dipilih.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "Referensi : Evans, J.D. (1996). Straightforward Statistics for the Behavioral Sciences.  "
    f"Dataset : vgsales.csv (Kaggle, Video Game Sales) — sampel {SAMPLE_N} records, random_state={SEED}."
)
