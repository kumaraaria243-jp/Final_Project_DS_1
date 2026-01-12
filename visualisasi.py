# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ======================
# LIGHT CSS (clean & KPI readable)
# ======================
def inject_light_css(hide_sidebar: bool = True):
    sidebar_css = """
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stAppViewContainer"] .main { margin-left: 0rem; }
    """ if hide_sidebar else ""

    st.markdown(
        f"""
        <style>
          .block-container {{ padding-top: 1.1rem; padding-bottom: 2.2rem; }}
          #MainMenu, footer, header {{ visibility: hidden; }}

          h1, h2, h3 {{ letter-spacing: -0.02em; }}
          .muted {{ color: #6b7280; font-size: 0.92rem; }}

          .stTabs [data-baseweb="tab"] {{ border-radius: 12px; }}

          /* Filter container */
          .filter-box {{
            background: #ffffff;
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 6px 14px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
          }}

          /* KPI row with vertical dividers */
          .kpi-row {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            background: #ffffff;
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 14px;
            box-shadow: 0 8px 18px rgba(0,0,0,0.06);
            overflow: hidden;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
          }}

          .kpi-item {{
            padding: 14px 16px;
            text-align: center;
            border-right: 1px solid rgba(0,0,0,0.08);
          }}
          .kpi-item:last-child {{
            border-right: none;
          }}

          .kpi-label {{
            font-size: 0.85rem;
            color: #6b7280;
            font-weight: 700;
            margin-bottom: 6px;
          }}

          .kpi-value {{
            font-size: 1.7rem;
            font-weight: 900;
            color: #111827;
            line-height: 1.1;
          }}

          {sidebar_css}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ======================
# FORMAT EURO (EU style: dot thousands)
# ======================
def euro(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return "-"
    return f"€{x:,.0f}".replace(",", ".")

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def _normalize_yesno_col(series: pd.Series) -> pd.Series:
    """
    Normalize various representations to 0/1.
    Supports: 1/0, True/False, "Yes"/"No", "true"/"false", etc.
    """
    if series.dtype == bool:
        return series.astype(int)

    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(int)

    s = series.astype(str).str.strip().str.lower()
    return s.map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}).fillna(0).astype(int)

# ======================
# DASHBOARD PAGE
# ======================
def chart():
    inject_light_css(hide_sidebar=True)

    df = load_data("Paris-Housing-Excel.xlsx")

    st.header("Paris Housing Dashboard")
    st.write(
        "Paris Housing Dashboard menyajikan analisis harga properti di Kota Paris berdasarkan karakteristik utama properti, "
        "seperti kategori, status bangunan, luas bangunan, serta fasilitas pendukung. Dashboard ini dirancang "
        "untuk membantu pengguna memahami pola harga dan melakukan perbandingan antar segmen properti "
        "melalui KPI, filter interaktif, dan visualisasi data yang terstruktur."
    )
    st.divider()

    # ======================
    # FILTER
    # ======================
    st.subheader("Filter Data")

    def opts(col):
        return ["All"] + sorted(df[col].dropna().unique().tolist()) if col in df.columns else ["All"]

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        category = st.selectbox("Category", opts("category"))
    with f2:
        building = st.selectbox("Status Bangunan", opts("building_status"))
    with f3:
        pool_filter = st.selectbox("Kolom Renang", ["All", "Yes", "No"])
    with f4:
        yard_filter = st.selectbox("Halaman", ["All", "Yes", "No"])

    # ======================
    # APPLY FILTER
    # ======================
    data = df.copy()

    if category != "All" and "category" in data.columns:
        data = data[data["category"] == category]

    if building != "All" and "building_status" in data.columns:
        data = data[data["building_status"] == building]

    if "hasPool" in data.columns:
        data["hasPool_norm"] = _normalize_yesno_col(data["hasPool"])
        if pool_filter == "Yes":
            data = data[data["hasPool_norm"] == 1]
        elif pool_filter == "No":
            data = data[data["hasPool_norm"] == 0]

    if "hasYard" in data.columns:
        data["hasYard_norm"] = _normalize_yesno_col(data["hasYard"])
        if yard_filter == "Yes":
            data = data[data["hasYard_norm"] == 1]
        elif yard_filter == "No":
            data = data[data["hasYard_norm"] == 0]

    st.caption(f"📊 Data setelah filter: **{len(data):,}**")

    if data.empty:
        st.warning("Tidak ada data sesuai filter.")
        return

    st.divider()

    # ======================
    # KPI (based on filtered data)
    # ======================
    st.subheader("KPI (Key Performance Indicators)")

    price = data["price"] if "price" in data.columns else pd.Series(dtype=float)
    sqm = data["squareMeters"] if "squareMeters" in data.columns else pd.Series(dtype=float)

    sqm_safe = sqm.replace(0, np.nan)
    ppm2 = (price / sqm_safe).replace([np.inf, -np.inf], np.nan)

    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi-item">
            <div class="kpi-label">Listings</div>
            <div class="kpi-value">{len(data):,}</div>
          </div>
          <div class="kpi-item">
            <div class="kpi-label">Avg Price</div>
            <div class="kpi-value">{euro(float(price.mean()))}</div>
          </div>
          <div class="kpi-item">
            <div class="kpi-label">Median Price</div>
            <div class="kpi-value">{euro(float(price.median()))}</div>
          </div>
          <div class="kpi-item">
            <div class="kpi-label">Avg Area (m²)</div>
            <div class="kpi-value">{float(sqm.mean()):,.1f}</div>
          </div>
          <div class="kpi-item">
            <div class="kpi-label">Median Price/m²</div>
            <div class="kpi-value">{euro(float(ppm2.median()))}</div>
          </div>
          <div class="kpi-item">
            <div class="kpi-label">Max Price</div>
            <div class="kpi-value">{euro(float(price.max()))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ======================
    # VISUALISASI
    # ======================
    st.subheader("Visualizations")

    t1, t2, t3, t4 = st.tabs(
        ["Area vs Price", "Price Distribution", "Price Boxplot", "Distribution"]
    )

    with t1:
        fig = px.scatter(
            data,
            x="squareMeters",
            y="price",
            trendline="ols",
            opacity=0.6,
            title="Area (m²) vs Price"
        )
        fig.update_yaxes(tickprefix="€", tickformat=",.0f")
        fig.update_traces(
            hovertemplate="Area: %{x} m²<br>Price: €%{y:,.0f}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig = px.histogram(
            data,
            x="price",
            nbins=40,
            marginal="box",
            title="Price Distribution"
        )
        fig.update_xaxes(tickprefix="€", tickformat=",.0f")
        fig.add_vline(x=float(data["price"].median()), line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        fig = px.box(
            data,
            y="price",
            points="outliers",
            title="Price Spread"
        )
        fig.update_yaxes(tickprefix="€", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        def _bar_dist(col: str, title: str):
            if col not in data.columns:
                st.info(f"Kolom `{col}` tidak ditemukan.")
                return
            vc = data[col].fillna("Unknown").astype(str).value_counts().reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(
                vc,
                x=col,
                y="count",
                text="count",
                title=title
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_title=None, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        def _donut_dist(col: str, title: str):
            if col not in data.columns:
                st.info(f"Kolom `{col}` tidak ditemukan.")
                return
            vc = data[col].fillna("Unknown").astype(str).value_counts().reset_index()
            vc.columns = [col, "count"]
            fig = px.pie(
                vc,
                names=col,
                values="count",
                hole=0.45,
                title=title
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        # category (bar)
        _bar_dist("category", "Distribution: Category (Bar)")

        # room_category (bar)
        _bar_dist("room_category", "Distribution: Room Category (Bar)")

        # floor_category (donut)
        _donut_dist("floor_category", "Distribution: Floor Category (Donut)")

        # garage_category removed as requested

    st.divider()

    # ======================
    # DATA TABLE (formatted)
    # ======================
    st.subheader("Filtered Data")
    st.write(
        "Bagian ini menampilkan data properti yang telah melalui proses penyaringan "
        "berdasarkan kriteria filter yang diterapkan pada dashboard. Data yang disajikan "
        "merupakan hasil akhir dari analisis dan digunakan sebagai dasar untuk melakukan evaluasi "
        "lebih lanjut terhadap temuan yang ditunjukkan pada visualisasi sebelumnya."
    )

    display_data = data.copy()

    # Add formatted Euro columns
    if "price" in display_data.columns:
        display_data["price (€)"] = display_data["price"].apply(euro)

    if "squareMeters" in display_data.columns and "price" in display_data.columns:
        ppm2_series = (display_data["price"] / display_data["squareMeters"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        display_data["price_per_m2 (€)"] = ppm2_series.apply(euro)

    # Remove helper columns if created
    hide_cols = [c for c in ["hasPool_norm", "hasYard_norm"] if c in display_data.columns]
    if hide_cols:
        display_data = display_data.drop(columns=hide_cols)

    st.dataframe(display_data, use_container_width=True, height=420)

# Run:
# streamlit run app.py
if __name__ == "__main__":
    chart()