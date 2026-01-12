from xml.parsers.expat import model
from matplotlib.pyplot import axes
import streamlit as st
import pandas as pd
import plotly.express as px
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import ConfusionMatrixDisplay
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression

def ml_model():
    df = pd.read_excel('Paris-Housing-Excel.xlsx')

    # 1. Membagi kolom numerik dan kategorik
    numbers = df.select_dtypes(include=['number']).columns
    categories = df.select_dtypes(exclude=['number']).columns

    # 2. Intoduction Machine Learning Paris Housing
    st.header('Introduction Machine Learning Paris Housing')
    col1, col2 = st.columns([6,4])
    with col1:
        link = "https://i.pinimg.com/1200x/cd/8f/42/cd8f42587b3df3cfe06686b32a41c120.jpg"
        st.image(link, caption="Pictures From Pinterest.com")
    with col2:
        st.write('Machine learning Housing Paris dapat digunakan untuk memprediksi harga properti di Paris dengan '\
                 'memanfaatkan data historis perumahan. Dataset housing Paris umumnya berisi '\
                 'informasi seperti luas bangunan, jumlah kamar, lokasi (arrondissement), usia bangunan, '\
                 'dan akses transportasi. Dengan pendekatan supervised learning, model machine learning dilatih untuk '
                 'mengenali pola hubungan antara karakteristik properti dan harga '\
                 'rumah. Hasil prediksi ini dapat membantu pembeli, penjual, maupun agen properti '\
                 'dalam mengambil keputusan yang lebih akurat dan berbasis data.')

    # 3. Deteksi dan penanganan outlier dengan (IQR Method)
    st.subheader('1. Deteksi dan Penanganan Outlier dengan IQR Method')
    Q1 = df[numbers].quantile(0.25)
    Q3 = df[numbers].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[~((df[numbers] < lower_bound) | (df[numbers] > upper_bound)).any(axis=1)]
    st.write(f"Jumlah data sebelum pembersihan: **:blue[{df.shape[0]} baris]**, setelah pembersihan outlier: **:blue[{df.shape[0]} baris]**")
    st.write('Berdasarkan hasil pemeriksaan outlier, jumlah data '':blue[sebelum dan setelah]'' proses'\
             ' '':blue[pembersihan]'' tetap '':blue[sebesar 10.000 baris]''. Hal ini menunjukkan bahwa tidak terdapat '\
             'nilai ekstrem yang memenuhi kriteria sebagai outlier, sehingga seluruh data dinilai '\
             'masih berada dalam rentang yang wajar. Oleh karena itu, tidak ada data yang '\
             'dieliminasi dan '':blue[dataset dapat digunakan secara keseluruhan pada tahap analisis]'' dan'\
             ' '':blue[pemodelan]'' selanjutnya.')

    # 4. Membaca Dataset Yang Digunakan
    df_select = df.copy()
    st.write('**Dataset yang digunakan (Paris Housing)**')
    st.dataframe(df.head())

    # 5. Memisahkan variabel Bebas(Didrop Beberapa) dan Terikat(Target)
    X = df_select.drop(['cityCode', 'cityPartRange', 'numPrevOwners', 'made', 'price', 'category',
                        'squareMeters_category', 'room_category', 'floor_category', 'owner_history_category',
                        'building_status', 'garage_category'], axis=1)
    Y = df_select['price']

    # 6. Correlation Heatmap untuk melihat korelasi linear antara kolom-kolom numerik
    st.subheader('2. Correlation Heatmap')
    st.write('Visualisasi ini digunakan untuk menggambarkan hubungan antar variabel numerik yang memengaruhi '\
             'harga properti di Paris. Visualisasi ini menunjukkan tingkat kekuatan dan arah korelasi antara '\
             'fitur seperti luas bangunan, jumlah kamar, usia bangunan, dan harga rumah.')

    # a. Visualisasi Correlation Heatmap
    corr = df_select[numbers].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    
    # b. Deskripsi Correlation Heatmap
    st.write('**Deskripsi Correlation Heatmap**')
    st.write("""
    - '':blue[Luas Bangunan adalah Penentu Harga Utama]'', dengan '':blue[korelasi 1 (sangat kuat) terhadap Harga]''.
        Ini menunjukkan bahwa harga properti hampir sepenuhnya mengikuti luas bangunan.
    - '':blue[Fitur Tambahan Tidak Menaikkan Harga Secara Linear]''.
        Fitur seperti hasPool, hasYard, hasStormProtector, hasStorageRoom, hasGuestRoom
        tidak menaikkan harga secara langsung.
    - Jumlah Ruangan & Lantai Bukan Faktor Dominan. 
        numberOfRooms dan floors menunjukkan korelasi lemah dengan harga.
    - Lokasi (cityCode, cityPartRange) Tidak Terlihat Kuat Secara Linear.
        Korelasi lokasi terhadap price sangat kecil.
    - '':blue[Dataset Minim Risiko Multikolinearitas]''.
        Hampir semua fitur tidak saling berkorelasi kuat.
    - Harga Properti Tampak “Oversimplified”
        '':blue[Satu variabel (squareMeters) sangat dominan dan Variabel lain hampir tidak berpengaruh]''.
    """)
    
    # 7. Mengecek Nilai VIF Setiap Kolom
    st.subheader('3. Mengecek Nilai VIF Setiap Kolom')
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])]
    st.dataframe(vif_data)

    # 8. Pembagian Data Latih dan Data Uji
    st.subheader('4. Membagi Data Latih dan Data Uji')
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
 
    # Introduction Data Train dan Data Test
    st.write('Pembagian data dilakukan dengan '':blue[proporsi 80% sebagai data latih (training set)]'' dan '':blue[20%]'''\
             ' '':blue[sebagai data uji (testing set)]''.')
    
    # Fitur Tab Menampilkan Variabel Train dan Test
    tab1, tab2 = st.tabs(["**Variabel Latih (Train)**", "**Variabel Uji (Test)**"]) 
    with tab1:
        st.write('Variabel X (train)')
        st.dataframe(X_train.head())
        st.write('Variabel Y (train)')
        st.dataframe(Y_train.head())
    with tab2:
        st.write('Variabel X (Test)')
        st.dataframe(X_test.head())
        st.write('Variabel Y (Test)')
        st.dataframe(Y_test.head())

    # 9. Normalisasi Dengan StandarScaler
    st.subheader('5. Normalisasi Dengan StandardScaler')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # a. Intodruction Normalisasi StandardScaler
    st.write('Penerapan normalisasi bertujuan untuk menyamakan skala antar variabel, sehingga '\
             'tidak ada fitur yang mendominasi proses pembentukan model.')

    # b. Visualisasi Data Setelah Normalisasi Menggunakan Histogram
    st.write("**:blue[Grafik Distribusi] X (Train) dan X (Test) :blue[Setelah StandardScaler]**")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # X_train_scaled
    ax[0].hist(X_train_scaled.flatten(), bins=30)
    ax[0].set_title("Distribusi X_train_scaled")
    ax[0].set_xlabel("Nilai")
    ax[0].set_ylabel("Frekuensi")
    # X_test_scaled
    ax[1].hist(X_test_scaled.flatten(), bins=30)
    ax[1].set_title("Distribusi X_test_scaled")
    ax[1].set_xlabel("Nilai")
    ax[1].set_ylabel("Frekuensi")
    st.pyplot(fig)

    # b. Deskripsi Histogram Setelah Normalisasi StandardScaler
    st.write('**Deskripsi Visualisasi Normalisasi StandardScaler**')
    st.write("""
    - Hasil visualisasi menunjukkan bahwa :blue[data latih dan data uji memiliki distribusi yang terpusat] di :blue[sekitar nilai nol],
        yang menandakan bahwa proses normalisasi telah berhasil mengubah data sehingga memiliki nilai rata-rata mendekati nol.
    - :blue[Distribusi data latih dan data uji yang relatif serupa] menunjukkan bahwa proses normalisasi pada data uji dilakukan
        menggunakan parameter yang diperoleh dari data latih, sehingga tidak terjadi kebocoran data (data leakage).
    - :blue[Rentang nilai pada kedua distribusi terlihat konsisten], yang mengindikasikan bahwa :blue[seluruh variabel independen telah
        berada pada skala yang seragam].
    """)

    # 10. Pemodelan Awal Linear Regresion
    st.subheader('6. Pemodelan Linear Regression')

    # a. Introduction Linear Regression
    st.write('Regresi linear merupakan salah satu '':blue[metode supervised learning]'' yang bertujuan untuk memprediksi nilai harga properti '\
             'berdasarkan hubungan linear antara variabel independen, seperti luas bangunan, jumlah kamar, dan lokasi, terhadap '\
             'variabel dependen dalam '':blue[memberikan gambaran awal mengenai pengaruh masing-masing variabel terhadap harga properti]'' di Paris.')

    # b. Membangun Model Linear Regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # c. Melihat Koefisien Masing2 Fitur
    coef_df = pd.DataFrame({
        'feature' : X.columns,
        'coefficient' : model.coef_
        })
    
    # d. Melihat Nilai Intercept
    st.write("Nilai Intercept : ", model.intercept_)

    # e. Ditampilkan di Streamlit
    st.write('**Menampilkan Koefisien Fitur**')
    st.dataframe(coef_df)

    # f. Evaluasi Model Linear Regression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    # prediksi
    Y_pred = model.predict(X_test)
    # MAE
    mae = mean_absolute_error(Y_test, Y_pred)
    # MAPE (Rumus Manual)
    mape = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100
    # RMSE
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    # R2 Score
    r2 = r2_score(Y_test, Y_pred)
    # Ditampilkan di Streamlit
    st.write('**Pemodelan Awal (Linear Regression)**')
    st.write("MAE  : ", mae)   
    st.write("MAPE : ", mape, "%")
    st.write("RMSE : ", rmse)
    st.write("R2   : ", r2)

    # 11. Melakukan Tunning Hyperparameter
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.model_selection import GridSearchCV

    # a. Introduction Tuning Hyperparameter
    st.subheader('7. Tuning Hyperparameter (Ridge dan Lasso)')
    st.write('Berfungsi untuk '':blue[mengoptimalkan kinerja model machine learning]'' dalam '':blue[memprediksi harga properti]''. '\
             'Proses ini dilakukan dengan menyesuaikan nilai hyperparameter agar model dapat mempelajari pola '\
             'hubungan antara karakteristik properti dan harga rumah secara lebih optimal.')

    # b. Fitur Colmns untuk menampilkan hasil Tuning Hyperparameter(Ridge dan Lasso)
    col1, col2 = st.columns([5,5])
    with col1:
        # Tuning Hyperparameter (Ridge)
        ridge = Ridge()
        #  dari 0.001 sampai 1000
        alphas = np.logspace(-3, 3, 20)
        param_grid = {'alpha' : alphas}

        grid = GridSearchCV(ridge, param_grid, cv = 10, scoring = 'neg_mean_squared_error')
        grid.fit(X_train_scaled, Y_train)

        st.write('**Tunning Hyperparameter '':blue[(Model Ridge)]''**')
        st.write("Best Alpha : ", grid.best_params_)
        st.write("Best Score : ", grid.best_score_)

    with col2:
        # Tuning Hyperparameter (Lasso) Menggunakan GridSearchCV
        from sklearn.model_selection import GridSearchCV
        lasso = Lasso()
        #  dari 0.001 sampai 1000
        alphas = np.logspace(-3, 3, 20)
        param_grid = {'alpha' : alphas}

        grid = GridSearchCV(lasso, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
        grid.fit(X_train_scaled, Y_train)
        st.write('**Tunning Hyperparameter '':blue[(Model Lasso)]''**')
        st.write("Best Alpha : ", grid.best_params_)
        st.write("Best Score : ", grid.best_score_)

    # 12. Pemodelan Ridge
    st.subheader('8. Model Ridge Regression')

    # a. Introduction Ridge
    st.write('Ridge Regression membantu menstabilkan koefisien model dengan cara memberi penalti pada nilai koefisien yang terlalu besar.')

    # b. Membangun Model Ridge
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha = 1.0)
    ridge.fit(X_train_scaled, Y_train)

    # c. Melihat Koefisien Masing-masing Fitur (setelah Ridge)
    coef_df_ridge = pd.DataFrame({
    'feature' : X.columns,
    'coefficient' : ridge.coef_
    })

    # d. Ditampilkan di Streamlit
    st.write('**Menampilkan Koefisien Fitur (setelah Ridge)**')
    st.dataframe(coef_df_ridge)

    # e. Prediksi Model (Ridge)
    Y_pred = ridge.predict(X_test_scaled)
    # MAE
    mae_ridge = mean_absolute_error(Y_test, Y_pred)
    # MAPE (Rumus Manual)
    mape_ridge = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100
    # RMSE
    rmse_ridge = np.sqrt(mean_squared_error(Y_test, Y_pred))
    # R2 Score
    r2_ridge = ridge.score(X_test_scaled, Y_test)
    # Ditampilkan di Streamlit
    st.write('**Pemodelan Ridge Regression**')
    st.write("MAE  : ", mae_ridge)
    st.write("MAPE : ", mape_ridge, "%")
    st.write("RMSE : ", rmse_ridge)
    st.write("R2   : ", r2_ridge)  

    # 13. Pemodelan Lasso
    st.subheader('8. Model Lasso Regression')

    # a. Introduction Lasso
    st.write('Lasso Regression membantu mengidentifikasi fitur-fitur yang paling berpengaruh terhadap harga rumah.')
    
    # b. Membangun Model Lasso
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha = 1.0)
    lasso.fit(X_train_scaled, Y_train)

    # c. Melihat Koefisien Masing-Masing Fitur setelah Lasso
    coef_df_lasso = pd.DataFrame({
    'feature' : X.columns,
    'coefficient' : lasso.coef_
    })

    # d. Ditampilkan di Streamlit
    st.write('**Menampilkan Koefisien Fitur (setelah Lasso)**')
    st.dataframe(coef_df_lasso)

    # e. Prediksi Model (Lasso)
    Y_pred = lasso.predict(X_test_scaled)
    # MAE
    mae_lasso = mean_absolute_error(Y_test, Y_pred)
    # MAPE (Manual Rumus)
    mape_lasso = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100
    # RMSE
    rmse_lasso = np.sqrt(mean_squared_error(Y_test, Y_pred))
    # R2 Score
    r2_lasso = lasso.score(X_test_scaled, Y_test)
    # Ditampilkan di Streamlit
    st.write('**Pemodelan Lasso Regression**')
    st.write("MAE  : ", mae_lasso)
    st.write("MAPE : ", mape_lasso, "%")
    st.write("RMSE : ", rmse_lasso)
    st.write("R2   : ", r2_lasso)

    # 14. Memilih Model Terbaik
    st.subheader('9. Evaluasi Nilai Alpha Terbaik (Ridge dan Lasso)')
    
    # a. Introduction Memilih Model Terbaik
    st.write('Evaluasi nilai alpha pada metode Ridge Regression dan Lasso Regression '':blue[bertujuan untuk '\
             'menentukan tingkat regularisasi yang optimal dalam memprediksi harga rumah]'' pada dataset Paris Housing.'\
             'Parameter alpha mengontrol besar penalti terhadap koefisien model, sehingga berpengaruh langsung terhadap '\
             'kompleksitas dan kemampuan generalisasi model. Melalui evaluasi ini, diperoleh nilai alpha terbaik yang '\
             'menghasilkan performa prediksi paling optimal.')
    
    # b. Membangun Model Terbaik Ridge dan Lasso dengan (Validasi RMSE)
    alphas = np.logspace(-3, 3, 20) # Alpha dari 0.001 - 1000
    ridge_rmse = []
    lasso_rmse = []
    for a in alphas :
        # Ridge
        ridge = Ridge(alpha = a , random_state = 42)
        ridge.fit(X_train_scaled, Y_train)
        Y_pred_ridge = ridge.predict(X_test_scaled)
        ridge_rmse.append(np.sqrt(mean_squared_error(Y_test, Y_pred_ridge)))

        # Lasso
        lasso = Lasso(alpha = a, max_iter = 10000, random_state = 42)
        lasso.fit(X_train_scaled, Y_train)
        Y_pred_lasso = lasso.predict(X_test_scaled)
        lasso_rmse.append(np.sqrt(mean_squared_error(Y_test, Y_pred_lasso)))

    # c. Validasi Dengan RMSE
    st.write('**Menampilkan Nilai RMSE untuk setiap Alpha**')
    # RMSE Ridge
    best_alpha_ridge = alphas[np.argmin(ridge_rmse)]
    best_rmse_ridge = min(ridge_rmse)
    # RMSE Lasso
    best_alpha_lasso = alphas[np.argmin(lasso_rmse)]
    best_rmse_lasso = min(lasso_rmse)

    # d. Ditampilkan di Streamlit
    st.write("""- Ridge Regression = Best Alpha : """, round(best_alpha_ridge, 3), """ | RMSE : """, round(best_rmse_ridge, 3))
    st.write("""- Lasso Regression = Best Alpha : """, round(best_alpha_lasso, 3), """ | RMSE : """, round(best_rmse_lasso, 3))

    
    # 15. Train Final Model (pakai best alpha)
    import joblib
    best_ridge = Ridge(alpha=best_alpha_ridge, random_state=42)
    best_ridge.fit(X_train_scaled, Y_train)

    best_lasso = Lasso(alpha=best_alpha_lasso, max_iter=10000, random_state=42)
    best_lasso.fit(X_train_scaled, Y_train)

    # 16. Save Model + Scaler + Feature Names
    feature_names = X.columns.tolist()
    joblib.dump(best_ridge, "ridge_best.pkl")
    joblib.dump(best_lasso, "lasso_best.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_names, "feature_names.pkl")
    # Ditampilkan di Streamlit
    st.success("Model berhasil disimpan")