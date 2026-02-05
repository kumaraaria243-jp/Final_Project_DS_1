import streamlit as st

st.title('🏠 Paris Housing Analysis and Prediction')
st.subheader('**Final Project Data Science**')
st.write('Jakarta, 7 Januari 2026')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['About Dataset', 
                            'Dashboards', 
                            'Machine Learning',
                            'Prediction App',
                            'Contact Me'])

with tab1:
    import about
    about.about_dataset()

with tab2:
    import visualisasi
    visualisasi.visualisasi_chart()

with tab3:
    import machine_learning
    machine_learning.ml_model()

with tab4:
    import prediction
    prediction.prediction_app()


with tab5:
    import kontak
    kontak.contact_me()
