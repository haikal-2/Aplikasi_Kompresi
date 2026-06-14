import streamlit as st
import io
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

# ==========================================
# FUNGSI KOMPRESI (BACK-END)
# ==========================================
def compress_jpeg_standard(image_pil, quality_value):
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG", quality=quality_value)
    buffer.seek(0)
    compressed_img = Image.open(buffer)
    compressed_size = len(buffer.getvalue())
    return compressed_img, compressed_size

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
    return compressed_img, len(buffer.getvalue())

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
    return compressed_img, len(buffer.getvalue())

# ==========================================
# ANTARMUKA PENGGUNA (GUI STREAMLIT)
# ==========================================
st.set_page_config(page_title="Studi Komparasi Kompresi Gambar", layout="wide")

st.title("Aplikasi Studi Komparasi Kompresi Gambar (JPEG)")
st.write("Silakan unggah satu atau beberapa gambar JPEG dan pilih algoritma kompresi untuk melihat perbandingannya.")

# 1. Area Upload Gambar (Bisa Banyak File & Mendukung Huruf Besar/Kecil)
uploaded_files = st.file_uploader(
    "Pilih Gambar JPEG (Bisa pilih lebih dari 1 file sekaligus)", 
    type=["jpg", "jpeg", "JPG", "JPEG"], 
    accept_multiple_files=True
)

# 2. Area Pengaturan (Sidebar)
st.sidebar.header("Pengaturan Kompresi")
algorithm = st.sidebar.selectbox("Pilih Algoritma Kompresi:", 
                                 ("JPEG Standard", "K-Means Clustering", "Singular Value Decomposition (SVD)"))

# Parameter dinamis berdasarkan algoritma yang dipilih
if algorithm == "JPEG Standard":
    param = st.sidebar.slider("Kualitas JPEG (Makin kecil makin terkompresi):", min_value=1, max_value=100, value=30)
elif algorithm == "K-Means Clustering":
    param = st.sidebar.slider("Jumlah Warna Dominan:", min_value=2, max_value=64, value=16)
else:
    param = st.sidebar.slider("Komponen Utama SVD (k):", min_value=10, max_value=200, value=50)

# 3. Tombol Eksekusi & Proses Looping untuk Banyak File
if uploaded_files: # Jika ada file yang diunggah
    if st.sidebar.button("Mulai Kompresi Semua Gambar"):
        
        # Lakukan perulangan untuk memproses setiap gambar yang di-upload
        for index, uploaded_file in enumerate(uploaded_files):
            st.write(f"### 🖼️ Memproses Gambar ke-{index+1}: **{uploaded_file.name}**")
            
            with st.spinner(f'Sedang mengompresi {uploaded_file.name}...'):
                # Baca gambar asli dan pastikan formatnya RGB (hilangkan transparansi)
                original_img = Image.open(uploaded_file).convert("RGB")
                original_size = len(uploaded_file.getvalue())  
                
                # Eksekusi algoritma yang dipilih
                if algorithm == "JPEG Standard":
                    compressed_img, compressed_size = compress_jpeg_standard(original_img, param)
                elif algorithm == "K-Means Clustering":
                    compressed_img, compressed_size = compress_kmeans(original_img, param)
                else:
                    compressed_img, compressed_size = compress_svd(original_img, param)
                
                # Kalkulasi Persentase
                reduction_percentage = ((original_size - compressed_size) / original_size) * 100
                
                # 4. Tampilan Hasil per Gambar (Kiri: Asli, Kanan: Kompresi)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Gambar Asli")
                    st.image(original_img, use_column_width=True)
                    st.info(f"**Ukuran Awal:** {original_size / 1024:.2f} KB")
                    
                with col2:
                    st.subheader("Hasil Kompresi")
                    st.image(compressed_img, use_column_width=True)
                    st.success(f"**Ukuran Sesudah:** {compressed_size / 1024:.2f} KB")
                    st.warning(f"**Persentase Pengurangan Ukuran:** {reduction_percentage:.2f}%")
            
            # Beri garis pembatas antar gambar agar rapi
            st.divider()
else:
    st.info("💡 Tips: Anda bisa memblok/memilih 10 file sekaligus di dalam folder saat menekan tombol Browse files.")