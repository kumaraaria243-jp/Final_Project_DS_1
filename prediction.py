import streamlit as st
import pandas as pd
import numpy as np
import joblib

def prediction_app():
    st.set_page_config(page_title="Paris Housing Prediction", layout="wide")
    st.header("Prediction App — Paris Housing")

    # ======================
    # LOAD MODEL & SCALER
    # ======================
    ridge_model = joblib.load("ridge_best.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")

    # ======================
    # LOAD DATASET (REFERENCE ONLY)
    # ======================
    df = pd.read_excel("Paris-Housing-Excel.xlsx")

    drop_cols = [
        "cityCode", "cityPartRange", "numPrevOwners", "made", "price", "category",
        "squareMeters_category", "room_category", "floor_category",
        "owner_history_category", "building_status", "garage_category"
    ]

    X_ref = df.drop(drop_cols, axis=1)

    # Pastikan numerik
    for c in X_ref.columns:
        if not pd.api.types.is_numeric_dtype(X_ref[c]):
            X_ref[c] = pd.to_numeric(X_ref[c], errors="coerce")

    # Handle missing value
    X_ref = X_ref.fillna(X_ref.median(numeric_only=True))
    stats = X_ref.describe().T

    # ======================
    # LABEL YANG LEBIH RAMAH
    # ======================
    label_map = {
        "squareMeters": "Luas Bangunan (m²)",
        "numberOfRooms": "Jumlah Kamar",
        "floors": "Jumlah Lantai",
        "basement": "Luas Basement (m²)",
        "attic": "Luas Loteng (m²)",
        "garage": "Luas Garasi (m²)",
        "hasYard": "Memiliki Halaman",
        "hasPool": "Memiliki Kolam Renang",
        "isNewBuilt": "Bangunan Baru",
        "hasStormProtector": "Pelindung Badai",
        "hasStorageRoom": "Ruang Penyimpanan",
        "hasGuestRoom": "Kamar Tamu",
    }

    # ======================
    # INPUT FORM
    # ======================
    st.subheader("Input Fitur Properti")
    st.write("**Menggunakan Model Ridge Regression**")

    with st.form("pred_form"):
        input_data = {}
        cols = st.columns(2)

        for i, col in enumerate(feature_names):
            if col not in X_ref.columns:
                st.warning(f"Kolom '{col}' tidak ditemukan pada dataset referensi.")
                continue

            label = label_map.get(col, col)
            uniq = X_ref[col].dropna().unique()

            with cols[i % 2]:
                # Fitur biner (0/1)
                if len(uniq) <= 2 and set(np.round(uniq, 0)).issubset({0, 1}):
                    pilihan = st.selectbox(label, ["Tidak", "Ya"])
                    input_data[col] = 1 if pilihan == "Ya" else 0

                # Fitur numerik
                else:
                    min_v = int(stats.loc[col, "min"])
                    max_v = int(stats.loc[col, "max"])
                    mean_v = int(stats.loc[col, "mean"])

                    val = st.number_input(
                        label,
                        min_value=min_v,
                        max_value=max_v,
                        value=mean_v,
                        step=1,
                        format="%d"
                    )
                    input_data[col] = int(val)

        submit = st.form_submit_button("Prediksi Harga")

    # ======================
    # PREDIKSI
    # ======================
    if submit:
        # Pastikan urutan fitur sesuai training
        input_df = pd.DataFrame(
            [[input_data.get(c, 0) for c in feature_names]],
            columns=feature_names
        )

        input_scaled = scaler.transform(input_df)
        pred = float(ridge_model.predict(input_scaled)[0])

        st.success(f"💶 Estimasi Harga Properti: **€ {pred:,.2f}**")

        with st.expander("Detail Input"):
            st.dataframe(input_df)


if __name__ == "__main__":
    prediction_app()
