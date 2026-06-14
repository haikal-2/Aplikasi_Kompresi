import streamlit as st
import io
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

# ==========================================
# FUNGSI KOMPRESI (BACK-END) + FITUR DOWNLOAD
# ==========================================
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
    """
    Fungsi ini menggabungkan ketiga algoritma secara berurutan:
    K-Means -> SVD -> JPEG Standard
    """
    # 1. K-Means
    img_np = np.array(image_pil)
    h, w, c = img_np.shape
    pixels = img_np.reshape(-1, c)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init=3)
    labels = kmeans.fit_predict(pixels)
    colors = kmeans.cluster_centers_.astype(np.uint8)
    compressed_pixels = colors[labels]
    compressed_np = compressed_pixels.reshape(h, w, c)
    img_k = Image.fromarray(compressed_np)

    # 2. SVD
    img_np_2 = np.array(img_k)
    compressed_channels = []
    for i in range(3):
        channel = img_np_2[:, :, i].astype(float)
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        compressed_channel = np.dot(U[:, :k_values], np.dot(np.diag(S[:k_values]), Vt[:k_values, :]))
        compressed_channels.append(compressed_channel)
    compressed_np_2 = np.stack(compressed_channels, axis=2)
    compressed_np_2 = np.clip(compressed_np_2, 0, 255).astype(np.uint8)
    img_svd = Image.fromarray(compressed_np_2)

    # 3. JPEG Standard
    buffer = io.BytesIO()
    img_svd.save(buffer, format="JPEG", quality=quality_value)
    buffer.seek(0)
    img_bytes = buffer.getvalue()
    final_img = Image.open(buffer)
    
    return final_img, len(img_bytes), img_bytes


# ==========================================
# ANTARMUKA PENGGUNA (GUI) - VERSI PRESENTASI + SVD + KOMBINASI
# ==========================================
st.set_page_config(page_title="Studi Komparasi 3 Algoritma", layout="wide", page_icon="⚖️")

st.title("⚖️ Studi Komparasi 3 Algoritma Kompresi (JPEG)")
st.write("Membandingkan performa individu: **JPEG Standard, K-Means, dan SVD**, serta **Kompresi Gabungan (Hybrid)** dari ketiganya.")

# 1. Area Upload Gambar
uploaded_files = st.file_uploader(
    "Unggah Minimal 10 Gambar JPEG di sini:", 
    type=["jpg", "jpeg", "JPG", "JPEG"], 
    accept_multiple_files=True
)

# 2. Area Pengaturan 3 Algoritma (Sidebar)
st.sidebar.header("⚙️ Pengaturan Parameter")

st.sidebar.markdown("**1. Algoritma JPEG Standard**")
jpeg_q = st.sidebar.slider("Tingkat Kualitas (%)", 1, 100, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("**2. Algoritma K-Means**")
kmeans_c = st.sidebar.slider("Jumlah Warna Dominan", 2, 64, 16)

st.sidebar.markdown("---")
st.sidebar.markdown("**3. Algoritma SVD**")
svd_k = st.sidebar.slider("Komponen Utama (k)", 10, 200, 50)

# 3. Proses Komparasi
if uploaded_files:
    if st.sidebar.button("🚀 Mulai Uji Komparasi", use_container_width=True):
        
        report_data = []
        st.write("### 📸 Perbandingan Visual per Gambar")
        
        for index, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"Gambar {index+1}: {uploaded_file.name}", expanded=True):
                with st.spinner(f'Mengompresi {uploaded_file.name}...'):
                    
                    # Baca gambar asli
                    original_img = Image.open(uploaded_file).convert("RGB")
                    size_ori = len(uploaded_file.getvalue())
                    
                    # Jalankan 3 Algoritma + 1 Kombinasi
                    img_jpeg, size_jpeg, byte_jpeg = compress_jpeg_standard(original_img, jpeg_q)
                    img_kmeans, size_kmeans, byte_kmeans = compress_kmeans(original_img, kmeans_c)
                    img_svd, size_svd, byte_svd = compress_svd(original_img, svd_k)
                    img_combo, size_combo, byte_combo = compress_combined(original_img, jpeg_q, kmeans_c, svd_k)
                    
                    # Tampilkan dalam 5 Kolom Sejajar
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.markdown("<h5 style='text-align: center;'>Asli</h5>", unsafe_allow_html=True)
                        st.image(original_img, use_column_width=True)
                        st.info(f"**Ukuran:** {size_ori/1024:.2f} KB")
                        
                    with col2:
                        st.markdown("<h5 style='text-align: center;'>1. JPEG</h5>", unsafe_allow_html=True)
                        st.image(img_jpeg, use_column_width=True)
                        pct = ((size_ori - size_jpeg) / size_ori) * 100
                        st.success(f"**Ukuran:** {size_jpeg/1024:.2f} KB")
                        st.warning(f"**Susut:** {pct:.2f}%")
                        st.download_button("⬇️ JPEG", byte_jpeg, f"JPEG_{uploaded_file.name}", "image/jpeg", key=f"d1_{index}")
                        
                    with col3:
                        st.markdown("<h5 style='text-align: center;'>2. K-Means</h5>", unsafe_allow_html=True)
                        st.image(img_kmeans, use_column_width=True)
                        pct = ((size_ori - size_kmeans) / size_ori) * 100
                        st.success(f"**Ukuran:** {size_kmeans/1024:.2f} KB")
                        st.warning(f"**Susut:** {pct:.2f}%")
                        st.download_button("⬇️ K-Means", byte_kmeans, f"KMEANS_{uploaded_file.name}", "image/jpeg", key=f"d2_{index}")
                        
                    with col4:
                        st.markdown("<h5 style='text-align: center;'>3. SVD</h5>", unsafe_allow_html=True)
                        st.image(img_svd, use_column_width=True)
                        pct = ((size_ori - size_svd) / size_ori) * 100
                        st.success(f"**Ukuran:** {size_svd/1024:.2f} KB")
                        st.warning(f"**Susut:** {pct:.2f}%")
                        st.download_button("⬇️ SVD", byte_svd, f"SVD_{uploaded_file.name}", "image/jpeg", key=f"d3_{index}")
                        
                    with col5:
                        st.markdown("<h5 style='text-align: center;'>🔥 Kombinasi (3)</h5>", unsafe_allow_html=True)
                        st.image(img_combo, use_column_width=True)
                        pct = ((size_ori - size_combo) / size_ori) * 100
                        st.success(f"**Ukuran:** {size_combo/1024:.2f} KB")
                        st.warning(f"**Susut:** {pct:.2f}%")
                        st.download_button("⬇️ Kombinasi", byte_combo, f"COMBO_{uploaded_file.name}", "image/jpeg", key=f"d4_{index}")
                        
                    # Simpan data untuk tabel
                    report_data.append({
                        "Nama File": uploaded_file.name,
                        "Asli (KB)": round(size_ori / 1024, 2),
                        "JPEG Standard (KB)": round(size_jpeg / 1024, 2),
                        "K-Means (KB)": round(size_kmeans / 1024, 2),
                        "SVD (KB)": round(size_svd / 1024, 2),
                        "Kombinasi (KB)": round(size_combo / 1024, 2)
                    })

        # 4. Tabel dan Grafik Ringkasan
        st.divider()
        st.write("### 📈 Rangkuman Performa 3 Algoritma & Kombinasi")
        
        df_report = pd.DataFrame(report_data)
        st.dataframe(df_report, use_container_width=True)
        
        st.write("#### Grafik Perbandingan Ukuran File")
        chart_data = df_report.set_index("Nama File")
        st.bar_chart(chart_data)
else:
    st.info("💡 Tips: Anda bisa memblok/memilih 10 file sekaligus di dalam folder saat menekan tombol Browse files.")
