import streamlit as st

def about_dataset():
    st.write('**About Paris Housing**')
    col1, col2= st.columns([5,5])

    with col1:
        link = "https://i.pinimg.com/1200x/d4/34/7d/d4347d2934950e3dda76e37e0b3f238e.jpg"
        st.image(link, caption="Housing Paris")

    with col2:
        st.write('Dataset Paris Housing berisi data mengenai properti perumahan di Paris, termasuk ' \
        'ukuran bangunan, jumlah ruangan, jumlah lantai, fasilitas rumah, serta status ' \
        'bangunan. Setiap data juga dilengkapi dengan informasi harga properti.' \
        'Selain itu, dataset ini juga memuat informasi tambahan seperti status bangunan ' \
        '(baru atau lama), jumlah pemilik sebelumnya, tahun pembangunan, dan kategori ' \
        'properti. '\
        'Setiap baris data merepresentasikan satu properti lengkap dengan harga ' \
        'jualnya, sehingga dataset ini sangat cocok digunakan untuk memahami kondisi ' \
        'pasar properti.'
        'Dataset Paris Housing dapat digunakan untuk analisis eksploratif data (EDA) guna '\
        'melihat pola dan hubungan antar variabel, serta untuk membangun model machine '\
        'learning dalam memprediksi harga rumah berdasarkan karakteristik yang '\
        'dimilikinya. Dataset ini bermanfaat bagi pelajar, peneliti, maupun praktisi yang '\
        'tertarik pada analisis data dan bidang properti.')