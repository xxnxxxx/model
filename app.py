import streamlit as st
import joblib
import os

# Fungsi untuk memuat file model dari direktori
def load_files_by_keyword(directory, keyword):
    return [f for f in os.listdir(directory) if keyword in f and f.endswith('.pkl')]

# Direktori tempat file .pkl disimpan
model_directory = "."

# Memuat daftar model
model_files = load_files_by_keyword(model_directory, "Model")

# Header aplikasi dengan responsivitas
st.markdown(
    """
    <div style="background-color:#1F4529;padding:10px;border-radius:10px;text-align:center">
        <h2 style="color:white;font-family:sans-serif;font-size:24px;">Aplikasi Prediksi Sentimen Mobil Hybrid</h2>
        <p style="color:white;font-size:16px;">Pilih model dan masukkan teks untuk prediksi</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Pilihan model
selected_model = st.selectbox("Pilih model:", model_files)

# Input teks dengan ukuran responsif
input_text = st.text_area(
    "Masukkan teks untuk prediksi",
    placeholder="Contoh: Mobil hybrid ramah lingkungan.",
    height=100,
)

# Tombol prediksi dengan jarak antar elemen
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
if st.button("Prediksi"):
    if input_text:
        try:
            # Memuat model yang dipilih
            svm_model = joblib.load(selected_model)

            # Secara otomatis menentukan vectorizer berdasarkan model yang dipilih
            vectorizer_name = selected_model.replace("Model", "vector").replace(".pkl", ".pkl")
            vectorizer_path = os.path.join(model_directory, vectorizer_name)

            if not os.path.exists(vectorizer_path):
                st.error(f"⚠️ Vectorizer untuk model '{selected_model}' tidak ditemukan.")
                st.stop()

            # Memuat vectorizer
            vectorizer = joblib.load(vectorizer_path)

            # Transformasi input menggunakan vectorizer
            input_vector = vectorizer.transform([input_text])

            # Prediksi
            prediction = svm_model.predict(input_vector)

            # Menentukan warna berdasarkan hasil prediksi
            if prediction[0].lower() == "positif":
                color = "green"
            elif prediction[0].lower() == "negatif":
                color = "red"
            else:
                color = "blue"

            # Menampilkan hasil prediksi dengan responsivitas
            st.markdown(
                f"""
                <div style="text-align:center;margin-top:20px;">
                    <h2 style="color:{color};font-size:22px;"> Prediksi Kategori: {prediction[0]} </h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("⚠️ Silakan masukkan teks untuk prediksi.")
