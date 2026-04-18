import os, json, joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Debtor Risk Management", page_icon="📉",
                   layout="wide", initial_sidebar_state="expanded")

BASE           = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PROCESSED = os.path.join(BASE, "data_processed")
MODEL_DIR      = os.path.join(BASE, "model_artifacts")

COLOUR_HIGH   = "#E24B4A"
COLOUR_MED    = "#EF9F27"
COLOUR_LOW    = "#639922"
COLOUR_ACCENT = "#185FA5"

FEATURE_LABELS = {
    "invoice_count":            "Total invoices outstanding",
    "avg_invoice_value_usd":    "Average invoice value (USD)",
    "total_outstanding_usd":    "Total amount owed (USD)",
    "recency_days":             "Days since last purchase",
    "receipt_count":            "Payments received (unallocated)",
    "unapplied_credit_abs":     "Unallocated payment amount",
    "has_unapplied_receipt":    "Has unallocated payments",
    "receipt_to_invoice_ratio": "Payment-to-invoice ratio",
    "debit_note_count":         "Debit notes raised",
    "total_debit_note_usd":     "Total debit note amount (USD)",
    "credit_note_count":        "Credit notes issued",
    "total_sales_value":        "Total lifetime purchases",
    "avg_invoice_value":        "Average purchase value",
    "active_months":            "Months actively purchasing",
    "purchase_frequency":       "Purchase frequency (per month)",
    "relationship_years":       "Years as a customer",
    "unique_products":          "Number of products purchased",
    "unique_categories":        "Product categories purchased",
    "category_concentration":   "Product diversity (HHI)",
    "behaviour_cluster":        "Customer behaviour segment",
}

_BANG = chr(33)
CSS = f"""
<style>
    [data-testid="stSidebar"] {{ background: #0f1f35; }}
    [data-testid="stSidebar"] * {{ color: #c9d6e3 {_BANG}important; }}
    [data-testid="stSidebar"] .stRadio label {{
        padding: 10px 14px; border-radius: 8px; cursor: pointer;
        font-size: 15px; display: flex {_BANG}important;
        align-items: center; width: 100%; margin: 0 {_BANG}important;
    }}
    [data-testid="stSidebar"] .stRadio label:hover {{ background: #1e3a5f; }}
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {{
        width: 14px {_BANG}important; height: 14px {_BANG}important;
        border-radius: 50% {_BANG}important;
        border: 2px solid #3d6a9e {_BANG}important;
        background: transparent {_BANG}important; flex-shrink: 0;
    }}
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] > div:first-child {{
        background: #E24B4A {_BANG}important; border-color: #E24B4A {_BANG}important;
    }}
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] {{ background: #1e3a5f; border-radius: 8px; }}
    .kpi-card {{ background:#f8f9fa; border-radius:10px; padding:16px; text-align:center; border:1px solid #e9ecef; }}
    .kpi-value {{ font-size:28px; font-weight:700; margin:0; }}
    .kpi-label {{ font-size:12px; color:#6c757d; margin:4px 0 0; }}
    .badge-high {{ background:#fdecea; color:#c0392b; padding:4px 12px; border-radius:12px; font-size:13px; font-weight:600; }}
    .badge-med  {{ background:#fef3e2; color:#d68910; padding:4px 12px; border-radius:12px; font-size:13px; font-weight:600; }}
    .badge-low  {{ background:#eafaf1; color:#1e8449; padding:4px 12px; border-radius:12px; font-size:13px; font-weight:600; }}
    .callout-danger  {{ background:#fdecea; border-left:4px solid #c0392b; padding:12px 16px; border-radius:6px; font-size:14px; color:#7b241c; margin-bottom:14px; }}
    .callout-warning {{ background:#fef9e7; border-left:4px solid #d68910; padding:12px 16px; border-radius:6px; font-size:14px; color:#7d6608; margin-bottom:14px; }}
    .callout-info    {{ background:#eaf3fb; border-left:4px solid #2471a3; padding:12px 16px; border-radius:6px; font-size:14px; color:#1a5276; margin-bottom:14px; }}
    .callout-success {{ background:#eafaf1; border-left:4px solid #1e8449; padding:12px 16px; border-radius:6px; font-size:14px; color:#1e8449; margin-bottom:14px; }}
    .section-header {{ font-size:13px; font-weight:600; color:#495057; text-transform:uppercase; letter-spacing:0.05em; border-bottom:1px solid #dee2e6; padding-bottom:6px; margin-bottom:12px; }}
    .formula-box {{ background:#f4f6f9; border:1px solid #d0d7e0; border-radius:6px; padding:10px 16px; font-family:monospace; font-size:13px; color:#2c3e50; margin:8px 0 12px; }}
    #MainMenu {{visibility:hidden;}} footer {{visibility:hidden;}} header {{visibility:hidden;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    risk_df      = pd.read_csv(os.path.join(DATA_PROCESSED, "final_risk_output.csv"))
    behaviour_df = pd.read_csv(os.path.join(DATA_PROCESSED, "behaviour_clusters.csv"))
    master_df    = pd.read_csv(os.path.join(DATA_PROCESSED, "master_features.csv"))
    shap_df      = pd.read_csv(os.path.join(DATA_PROCESSED, "shap_values_all_customers.csv"))
    with open(os.path.join(DATA_PROCESSED, "kpi_summary.json")) as f:
        kpi = json.load(f)
    return risk_df, behaviour_df, master_df, shap_df, kpi


@st.cache_resource
def load_model():
    model        = joblib.load(os.path.join(MODEL_DIR, "rf_risk_band_model.pkl"))
    le           = joblib.load(os.path.join(MODEL_DIR, "risk_band_label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    import shap
    explainer = shap.TreeExplainer(model)
    return model, le, feature_cols, explainer


risk_df, behaviour_df, master_df, shap_df, kpi = load_data()
model, le, feature_cols, explainer = load_model()


def badge_html(bucket):
    if bucket == "High Risk":   return '<span class="badge-high">&#9888; High Risk</span>'
    if bucket == "Medium Risk": return '<span class="badge-med">&#9684; Medium Risk</span>'
    return '<span class="badge-low">&#10003; Low Risk</span>'


def gauge_figure(score, bucket):
    colour = COLOUR_HIGH if bucket == "High Risk" else COLOUR_MED if bucket == "Medium Risk" else COLOUR_LOW
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": " / 100", "font": {"size": 36, "color": colour}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": colour, "thickness": 0.25},
               "steps": [{"range": [0,30], "color": "#eafaf1"},
                         {"range": [30,70], "color": "#fef3e2"},
                         {"range": [70,100], "color": "#fdecea"}],
               "threshold": {"line": {"color": "#2c3e50", "width": 3}, "thickness": 0.8, "value": score}},
    ))
    fig.update_layout(height=230, margin=dict(t=20,b=0,l=20,r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family":"sans-serif"})
    return fig


def plotly_defaults(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="sans-serif", size=12), margin=dict(t=30,b=30,l=10,r=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.2)", zeroline=False)
    return fig


def profile_row(label, value):
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
        f"border-bottom:1px solid rgba(200,200,200,0.15);'>"
        f"<span style='color:#8a9bb0;font-size:13px;'>{label}</span>"
        f"<span style='font-size:13px;font-weight:500;'>{value}</span></div>",
        unsafe_allow_html=True)


def friendly_label(f):
    return FEATURE_LABELS.get(f, f.replace("_", " ").title())


risk_df["cuscode"]         = risk_df["cuscode"].astype(str).str.replace(",","").str.strip()
behaviour_df["customerid"] = behaviour_df["customerid"].astype(str)
master_df["customerid"]    = master_df["customerid"].astype(str)
shap_df["customerid"]      = shap_df["customerid"].astype(str)
risk_df["customerid"]      = risk_df["customerid"].astype(str)

_total     = kpi["total_customers"]
_high_n    = kpi["high_risk_count"]
_med_n     = kpi["medium_risk_count"]
_high_pct  = _high_n / _total * 100
_med_pct   = _med_n  / _total * 100
_high_usd  = kpi.get("high_risk_outstanding_usd", 0)
_total_usd = kpi["total_outstanding_usd"]
_high_usd_pct = (_high_usd / _total_usd * 100) if _total_usd else 0
_data_as_of   = kpi.get("data_as_of", "N/A")

# SIDEBAR 
with st.sidebar:
    st.markdown(f"""
        <div style='padding:18px 12px 10px;text-align:center;'>
          <svg viewBox="0 0 120 68" xmlns="http://www.w3.org/2000/svg"
               style="width:110px;display:block;margin:0 auto 10px;">
            <path d="M10,60 A50,50 0 0,1 110,60" fill="none" stroke="#1e3a5f" stroke-width="10" stroke-linecap="round"/>
            <path d="M10,60 A50,50 0 0,1 60,10"  fill="none" stroke="#639922" stroke-width="10" stroke-linecap="round"/>
            <path d="M60,10 A50,50 0 0,1 110,60" fill="none" stroke="#E24B4A" stroke-width="10" stroke-linecap="round"/>
            <line x1="60" y1="60" x2="32" y2="22" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="60" cy="60" r="4" fill="#ffffff"/>
            <text x="6"  y="72" font-size="8" fill="#639922" font-family="sans-serif">LOW</text>
            <text x="90" y="72" font-size="8" fill="#E24B4A" font-family="sans-serif">HIGH</text>
          </svg>
          <div style='font-size:13px;font-weight:600;color:#e8edf2;letter-spacing:0.04em;margin-bottom:2px;'>Debtor Risk Management</div>
          <div style='font-size:10px;color:#5a7a9a;'>Credit Intelligence System</div>
        </div>
        <div style='height:1px;background:rgba(255,255,255,0.08);margin:6px 0 14px;'></div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation",
        ["📊  Risk Overview", "🔍  Customer Explorer", "👥  Segments & Exposure"])

    st.markdown("<div style='height:1px;background:rgba(255,255,255,0.08);margin:14px 0 12px;'></div>",
                unsafe_allow_html=True)
    st.markdown(f"""
        <div style='font-size:11px;color:#8a9bb0;line-height:2;padding:0 4px;'>
          <div style='font-size:10px;font-weight:600;color:#5a7a9a;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Portfolio at a glance</div>
          <div style='display:flex;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.05);padding-bottom:4px;margin-bottom:4px;'>
            <span>Total accounts</span><span style='color:#c9d6e3;font-weight:600;'>{_total:,}</span>
          </div>
          <div style='display:flex;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.05);padding-bottom:4px;margin-bottom:4px;'>
            <span style='color:{COLOUR_HIGH};'>High risk</span>
            <span style='color:{COLOUR_HIGH};font-weight:600;'>{_high_n:,} ({_high_pct:.0f}%)</span>
          </div>
          <div style='display:flex;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.05);padding-bottom:4px;margin-bottom:4px;'>
            <span style='color:{COLOUR_MED};'>Medium risk</span>
            <span style='color:{COLOUR_MED};font-weight:600;'>{_med_n:,} ({_med_pct:.0f}%)</span>
          </div>
          <div style='display:flex;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.05);padding-bottom:4px;margin-bottom:8px;'>
            <span>Total outstanding</span><span style='color:#c9d6e3;font-weight:600;'>${_total_usd/1e6:.1f}M</span>
          </div>
          <div style='font-size:10px;color:#3d5a73;'>Data as of {_data_as_of}</div>
        </div>
    """, unsafe_allow_html=True)


# PAGE 1: RISK OVERVIEW 
if page == "📊  Risk Overview":
    st.title("Risk Intelligence Overview")
    st.markdown(
        f"<div class='callout-danger'>&#9888;&nbsp;<strong>${_high_usd/1e6:.1f}M of outstanding debt is in accounts flagged as high risk.</strong>"
        f"&nbsp;This is {_high_usd_pct:.1f}% of the total portfolio and requires immediate attention from the credit team.</div>",
        unsafe_allow_html=True)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    def kpi_card(col, value, label, colour="#2c3e50"):
        col.markdown(f"<div class='kpi-card'><p class='kpi-value' style='color:{colour};'>{value}</p>"
                     f"<p class='kpi-label'>{label}</p></div>", unsafe_allow_html=True)
    kpi_card(k1, f"{_total:,}", "Total customers")
    kpi_card(k2, f"{_high_n:,}", f"High risk ({_high_pct:.0f}%)", COLOUR_HIGH)
    kpi_card(k3, f"{_med_n:,}",  f"Medium risk ({_med_pct:.0f}%)", COLOUR_MED)
    kpi_card(k4, f"{kpi['low_risk_count']:,}", f"Low risk ({kpi['low_risk_count']/_total*100:.0f}%)", COLOUR_LOW)
    kpi_card(k5, f"${_total_usd/1e6:.1f}M", "Total outstanding (USD)", COLOUR_ACCENT)
    kpi_card(k6, f"{kpi.get('avg_overdue_days',0):.0f}d", "Avg overdue days", "#6c757d")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Portfolio Summary", "Global Exposure Map"])

    with tab1:
        col_a, col_b = st.columns([1, 1.6])
        with col_a:
            st.markdown("<div class='section-header'>Risk breakdown</div>", unsafe_allow_html=True)
            donut_df = risk_df["risk_bucket"].value_counts().reset_index()
            donut_df.columns = ["Risk Bucket", "Count"]
            fig_donut = px.pie(donut_df, names="Risk Bucket", values="Count", hole=0.55,
                color="Risk Bucket",
                color_discrete_map={"High Risk":COLOUR_HIGH,"Medium Risk":COLOUR_MED,"Low Risk":COLOUR_LOW})
            fig_donut.update_traces(textposition="outside", textinfo="percent+label")
            fig_donut.update_layout(height=300, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                                     margin=dict(t=20,b=20,l=10,r=10))
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_b:
            st.markdown("<div class='section-header'>Outstanding balance by risk level (USD)</div>",
                        unsafe_allow_html=True)
            exposure_df = risk_df.groupby("risk_bucket").agg(
                outstanding_usd=("total_outstanding_usd","sum"),
                account_count=("customerid","count")).reset_index()
            exposure_df["label"] = exposure_df.apply(
                lambda r: f"${r['outstanding_usd']/1e6:.1f}M  ({int(r['account_count'])} accounts)", axis=1)
            max_val = exposure_df["outstanding_usd"].max()
            fig_bar = px.bar(exposure_df, x="risk_bucket", y="outstanding_usd", color="risk_bucket",
                color_discrete_map={"High Risk":COLOUR_HIGH,"Medium Risk":COLOUR_MED,"Low Risk":COLOUR_LOW},
                text="label", labels={"outstanding_usd":"Outstanding (USD)","risk_bucket":""})
            fig_bar.update_traces(textposition="outside")
            plotly_defaults(fig_bar)
            fig_bar.update_layout(height=300, showlegend=False, yaxis=dict(range=[0, max_val*1.3]))
            st.plotly_chart(fig_bar, use_container_width=True)

        high_risk_df = risk_df[risk_df["risk_bucket"] == "High Risk"].copy()
        dl_cols = [c for c in ["cuscode","customername","customer","country","risk_score",
                               "risk_bucket","total_outstanding_usd","max_overdue_days",
                               "cluster_name","top_category"] if c in high_risk_df.columns]
        st.download_button("Download High Risk Account List (CSV)",
            data=high_risk_df[dl_cols].sort_values("risk_score",ascending=False).to_csv(index=False).encode(),
            file_name="high_risk_accounts.csv", mime="text/csv")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Top 10 accounts requiring attention</div>",
                    unsafe_allow_html=True)
        name_col = next((c for c in ["customername","customer","name"] if c in risk_df.columns), None)
        top10    = risk_df.sort_values("risk_score", ascending=False).head(10).copy()
        shap_cols_all = [c for c in shap_df.columns if c.startswith("shap_")]
        if shap_cols_all:
            top10_shap = top10.merge(shap_df[["customerid"]+shap_cols_all], on="customerid", how="left")
            def top_factor(row):
                vals = row[shap_cols_all]
                return "N/A" if vals.isna().all() else friendly_label(vals.astype(float).idxmax().replace("shap_",""))
            top10["Top Risk Factor"] = top10_shap.apply(top_factor, axis=1)
        dcols = ["cuscode"] + ([name_col] if name_col else [])
        for c in ["country","cluster_name","risk_score","risk_bucket","total_outstanding_usd",
                  "max_overdue_days","top_category","Top Risk Factor"]:
            if c in top10.columns: dcols.append(c)
        top10_d = top10[dcols].copy().rename(columns={
            "cuscode":"Code", "country":"Country", "cluster_name":"Segment",
            "risk_score":"Risk Score", "risk_bucket":"Risk Level",
            "total_outstanding_usd":"Outstanding", "max_overdue_days":"Max Overdue Days",
            "top_category":"Product Category", **({"customername":"Customer Name"} if name_col=="customername" else
                                                   {"customer":"Customer Name"} if name_col=="customer" else {})})
        if "Outstanding" in top10_d.columns:
            top10_d["Outstanding"] = top10_d["Outstanding"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top10_d, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("<div class='section-header'>Outstanding exposure by country</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='callout-info'>Bar length = total outstanding balance. "
            "Colour = average risk score — <strong>a shorter red bar can be more urgent than a longer green one.</strong> "
            "Use both together to prioritise collection efforts.</div>", unsafe_allow_html=True)
        if "country" in risk_df.columns:
            country_df = risk_df.groupby("country").agg(
                total_outstanding=("total_outstanding_usd","sum"),
                avg_risk_score=("risk_score","mean"),
                customers=("customerid","count"),
                high_risk=("risk_bucket", lambda x: (x=="High Risk").sum())).reset_index()
            country_df["avg_risk_score"] = country_df["avg_risk_score"].round(1)

            fig_map = px.scatter_geo(country_df, locations="country", locationmode="country names",
                size="total_outstanding", color="avg_risk_score",
                color_continuous_scale=["#27ae60","#f39c12","#e74c3c"], range_color=[0,100],
                hover_name="country",
                hover_data={"customers":True,"total_outstanding":":,.0f","avg_risk_score":True,"high_risk":True},
                labels={"total_outstanding":"Outstanding (USD)","avg_risk_score":"Risk Score",
                        "high_risk":"High Risk Accounts","customers":"Accounts"},
                size_max=50, projection="natural earth")
            fig_map.update_traces(marker=dict(sizemode="area",
                sizeref=2.*country_df["total_outstanding"].max()/(50.**2),
                sizemin=6, opacity=0.82, line=dict(width=1,color="white")))
            fig_map.update_layout(height=480, paper_bgcolor="white", plot_bgcolor="white",
                geo=dict(projection_type="natural earth", showframe=False, resolution=50,
                    showcoastlines=True, coastlinecolor="#aab4be",
                    showland=True, landcolor="#f0f2f5",
                    showocean=True, oceancolor="#d6e8f7",
                    showcountries=True, countrycolor="#b0bec5", bgcolor="white",
                    lataxis_range=[-58,82], lonaxis_range=[-170,180]),
                coloraxis_colorbar=dict(title="Risk Score", len=0.55, thickness=14, outlinewidth=0),
                margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("Bubble size = outstanding balance. Colour = avg risk score. Hover for details.")

            top_c = country_df.nlargest(10,"total_outstanding").copy()
            top_c["bar_label"] = top_c.apply(
                lambda r: f"${r['total_outstanding']/1e6:.1f}M  |  Risk: {r['avg_risk_score']:.0f}/100", axis=1)
            fig_cc = px.bar(top_c, x="total_outstanding", y="country", orientation="h",
                color="avg_risk_score",
                color_continuous_scale=["#27ae60","#f39c12","#e74c3c"], range_color=[0,100],
                text="bar_label",
                labels={"total_outstanding":"Outstanding (USD)","country":"","avg_risk_score":"Avg Risk Score"})
            fig_cc.update_traces(textposition="outside")
            plotly_defaults(fig_cc)
            fig_cc.update_layout(height=360, yaxis={"categoryorder":"total ascending"},
                                  xaxis=dict(range=[0, top_c["total_outstanding"].max()*1.5]))
            st.plotly_chart(fig_cc, use_container_width=True)
        else:
            st.info("Country data not available.")


# PAGE 2: CUSTOMER EXPLORER
elif page == "🔍  Customer Explorer":
    st.title("Customer Account Review")
    st.markdown("<div class='callout-info'>Select any customer to view their risk level, "
                "the key factors behind their score, and their account and purchase history.</div>",
                unsafe_allow_html=True)

    name_col = next((c for c in ["customername","customer","name"] if c in risk_df.columns), None)
    if name_col:
        risk_df["_label"] = (risk_df["cuscode"].astype(str).str.replace(",","").str.strip()
                             + "  —  " + risk_df[name_col].astype(str))
    else:
        risk_df["_label"] = risk_df["cuscode"].astype(str).str.replace(",","").str.strip()

    selected = st.selectbox("Search by customer name or code",
                            risk_df.sort_values("risk_score", ascending=False)["_label"].tolist(), index=0)
    cust    = risk_df[risk_df["_label"] == selected].iloc[0]
    cust_id = cust["customerid"]

    tab_a, tab_b = st.tabs(["Account Risk Summary", "Purchase History"])

    with tab_a:
        col_profile, col_gauge = st.columns([1,1])

        with col_profile:
            st.markdown("<div class='section-header'>Account details</div>", unsafe_allow_html=True)
            if name_col and name_col in cust.index: profile_row("Customer name", cust[name_col])
            profile_row("Account code", str(cust["cuscode"]).replace(",","").strip())
            if "country" in cust.index and pd.notna(cust.get("country")): profile_row("Country", cust["country"])
            if "term" in cust.index and pd.notna(cust.get("term")):       profile_row("Payment terms", cust["term"])
            if "cluster_name" in cust.index:                               profile_row("Customer type", cust["cluster_name"])
            if "top_category" in cust.index and pd.notna(cust.get("top_category")): profile_row("Primary product", cust["top_category"])
            profile_row("Invoices outstanding", int(cust["invoice_count"]))
            profile_row("Longest overdue",  f"{int(cust['max_overdue_days'])} days")
            profile_row("Average overdue",  f"{cust['avg_overdue_days']:.0f} days")
            profile_row("Balance outstanding", f"${cust['total_outstanding_usd']:,.0f} USD")
            profile_row("Payments on account", int(cust["receipt_count"]))
            profile_row("Debit notes raised",  int(cust["debit_note_count"]))

        with col_gauge:
            bucket = cust["risk_bucket"]
            score  = cust["risk_score"]
            st.markdown("<div class='section-header'>Payment risk assessment</div>", unsafe_allow_html=True)
            st.plotly_chart(gauge_figure(score, bucket), use_container_width=True)
            st.markdown(f"<div style='text-align:center;margin:-6px 0 6px;'>{badge_html(bucket)}</div>",
                        unsafe_allow_html=True)

            _prob_high = round(cust["prob_high"]*100, 1)
            _prob_med  = round(cust["prob_medium"]*100, 1)
            _prob_low  = round(cust["prob_low"]*100, 1)
            st.markdown(
                f"<div class='formula-box'>"
                f"<strong>Risk Score: {score:.0f} / 100</strong><br>"
                f"Probability of payment default: <strong>{_prob_high:.1f}%</strong><br>"
                f"Probability of medium delay: {_prob_med:.1f}%&nbsp;|&nbsp;"
                f"On-time payment: {_prob_low:.1f}%"
                f"</div>", unsafe_allow_html=True)

            if bucket == "High Risk":
                st.markdown("<div class='callout-danger'>This account shows multiple indicators of payment difficulty. "
                            "Immediate review by the credit team is recommended.</div>", unsafe_allow_html=True)
            elif bucket == "Medium Risk":
                st.markdown("<div class='callout-warning'>This account shows some early warning signals. "
                            "Monitor closely and consider proactive outreach.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='callout-success'>This account is currently in good standing "
                            "with low risk indicators.</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            prob_df = pd.DataFrame({"Level":["High","Medium","Low"],
                "Probability":[_prob_high, _prob_med, _prob_low]})
            fig_prob = px.bar(prob_df, x="Level", y="Probability", color="Level",
                color_discrete_map={"High":COLOUR_HIGH,"Medium":COLOUR_MED,"Low":COLOUR_LOW},
                text="Probability",
                labels={"Probability":"Model confidence (%)","Level":"Risk Level"},
                title="Model confidence by risk class")
            fig_prob.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            plotly_defaults(fig_prob)
            fig_prob.update_layout(height=220, showlegend=False, yaxis=dict(range=[0,130]))
            st.plotly_chart(fig_prob, use_container_width=True)
            st.caption("How confident the model is that this account falls into each risk class.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Key factors driving this risk assessment</div>",
                    unsafe_allow_html=True)
        cust_shap = shap_df[shap_df["customerid"] == cust_id]

        def render_shap_chart(shap_series):
            shap_series  = shap_series.sort_values(key=abs, ascending=False).head(8)
            total_abs    = shap_series.abs().sum()
            pct_influence = (shap_series.abs()/total_abs*100).round(1) if total_abs > 0 else shap_series.abs()
            plot_df = pd.DataFrame({
                "factor":    [friendly_label(f) for f in shap_series.index],
                "impact":    shap_series.values,
                "pct":       pct_influence.values,
                "direction": ["Increases risk" if v > 0 else "Reduces risk" for v in shap_series.values],
            })
            plot_df["label"] = plot_df.apply(
                lambda r: f"{r['pct']:.0f}% — {'pushes score up' if r['impact']>0 else 'pulls score down'}",
                axis=1)
            fig = px.bar(plot_df, x="pct", y="factor", orientation="h", color="direction",
                color_discrete_map={"Increases risk":COLOUR_HIGH,"Reduces risk":COLOUR_LOW},
                text="label", labels={"pct":"% of total risk influence","factor":""})
            fig.update_traces(textposition="outside", textfont_size=11)
            plotly_defaults(fig)
            fig.update_layout(height=360, yaxis={"categoryorder":"total ascending"},
                legend_title_text="",
                xaxis=dict(title="Percentage of total risk influence (%)",
                           range=[0, pct_influence.max()*1.7]))
            return fig, plot_df

        if not cust_shap.empty:
            s_cols = [c for c in cust_shap.columns if c.startswith("shap_")]
            sv = cust_shap[s_cols].iloc[0]
            sv.index = [c.replace("shap_","") for c in sv.index]
            fig_shap, shap_plot = render_shap_chart(sv)

            top_driver = shap_plot[shap_plot["impact"]>0].iloc[0] if shap_plot[shap_plot["impact"]>0].shape[0]>0 else None
            if top_driver is not None:
                st.markdown(
                    f"<div class='callout-danger' style='margin-bottom:8px;'>"
                    f"<strong>Primary risk driver:</strong> {top_driver['factor']} — this accounts for "
                    f"{top_driver['pct']:.0f}% of the total risk signal for this account.</div>",
                    unsafe_allow_html=True)

            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Each bar shows how much one factor contributed to this customer's risk score as a percentage "
                       "of total influence. Red = pushes score up. Green = pulls it down. Longer bar = stronger evidence.")

            pos_factors = shap_plot[shap_plot["impact"]>0].head(3)["factor"].tolist()
            neg_factors = shap_plot[shap_plot["impact"]<0].head(1)["factor"].tolist()
            if pos_factors:
                ft = ", ".join([f"<strong>{f}</strong>" for f in pos_factors])
                note = f"The primary concerns for this account are: {ft}."
                if neg_factors:
                    note += f" Partially offset by a positive signal from <strong>{neg_factors[0]}</strong>."
                st.markdown(f"<div class='callout-info'>{note}</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Analysing this account..."):
                try:
                    cm = master_df[master_df["customerid"]==cust_id]
                    if not cm.empty:
                        sv2 = explainer.shap_values(cm[feature_cols].values)
                        hi  = list(le.classes_).index("High")
                        arr = sv2[hi][0] if isinstance(sv2, list) else sv2[0,:,hi]
                        fig_shap, _ = render_shap_chart(pd.Series(arr, index=feature_cols))
                        st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not complete analysis: {e}")

        col_inv, col_doc = st.columns(2)
        with col_inv:
            st.markdown("<div class='section-header'>Invoice age profile</div>", unsafe_allow_html=True)
            aging_cols = ["outstanding_0_30","outstanding_31_60","outstanding_61_90","outstanding_91_plus"]
            if all(c in cust.index for c in aging_cols):
                inv_data = pd.DataFrame({
                    "Age band": ["Current (0-30d)","Late (31-60d)","Very late (61-90d)","Severely overdue (91+d)"],
                    "Outstanding (USD)": [float(cust[c]) for c in aging_cols]})
            else:
                od = int(cust["max_overdue_days"])
                inv_data = pd.DataFrame({
                    "Age band": ["Current (0-30d)","Late (31-60d)","Very late (61-90d)","Severely overdue (91+d)"],
                    "Outstanding (USD)": [max(0,30-min(od,30)), max(0,min(od,60)-30),
                                          max(0,min(od,90)-60), max(0,od-90)]})
            fig_inv = px.bar(inv_data, x="Age band", y="Outstanding (USD)", color="Age band",
                color_discrete_sequence=[COLOUR_LOW,COLOUR_MED,"#FF7043",COLOUR_HIGH],
                text=inv_data["Outstanding (USD)"].apply(lambda x: f"${x:,.0f}"))
            fig_inv.update_traces(textposition="outside")
            plotly_defaults(fig_inv)
            max_inv = inv_data["Outstanding (USD)"].max()
            fig_inv.update_layout(height=240, showlegend=False, yaxis=dict(range=[0,max(max_inv*1.3,1)]))
            st.plotly_chart(fig_inv, use_container_width=True)
            st.caption("Outstanding balance split by how long each invoice has been unpaid.")

        with col_doc:
            st.markdown("<div class='section-header'>Account activity breakdown</div>", unsafe_allow_html=True)
            doc_df = pd.DataFrame({
                "Document type": ["Invoices raised","Payments received","Debit notes","Credit notes"],
                "Count": [int(cust["invoice_count"]), int(cust["receipt_count"]),
                          int(cust["debit_note_count"]), int(cust.get("credit_note_count",0))]})
            fig_doc = px.bar(doc_df, x="Document type", y="Count", color="Document type",
                color_discrete_sequence=[COLOUR_ACCENT,COLOUR_LOW,COLOUR_HIGH,COLOUR_MED], text="Count")
            fig_doc.update_traces(textposition="outside")
            plotly_defaults(fig_doc)
            fig_doc.update_layout(height=240, showlegend=False,
                                   yaxis=dict(range=[0, max(doc_df["Count"].max()*1.35, 1)]))
            st.plotly_chart(fig_doc, use_container_width=True)

    with tab_b:
        cust_beh = behaviour_df[behaviour_df["customerid"]==cust_id]
        if not cust_beh.empty:
            cb = cust_beh.iloc[0]
            b1,b2,b3,b4 = st.columns(4)
            b1.metric("Total purchases",      f"${cb['total_sales_value']:,.0f}")
            b2.metric("Active months",        int(cb["active_months"]))
            b3.metric("Products ordered",     int(cb["unique_products"]))
            b4.metric("Avg orders per month", f"{cb['purchase_frequency']:.2f}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Purchasing activity overview</div>", unsafe_allow_html=True)

            first_dt      = pd.to_datetime(cb["first_purchase"])
            last_dt       = pd.to_datetime(cb["last_purchase"])
            snap_dt       = pd.Timestamp(_data_as_of) if _data_as_of not in ("N/A",) else pd.Timestamp("today")
            active_days   = max((last_dt - first_dt).days, 1)
            inactive_days = max((snap_dt - last_dt).days, 0)
            total_days    = active_days + inactive_days
            active_pct    = round(active_days / total_days * 100)
            inactive_pct  = 100 - active_pct
            inact_col     = "#c0392b" if inactive_days > 365 else "#d68910" if inactive_days > 180 else "#8a9bb0"

            tl1,tl2,tl3 = st.columns(3)
            tl1.markdown(f"<div class='kpi-card'><p class='kpi-value' style='color:{COLOUR_LOW};'>"
                         f"{first_dt.strftime('%b %Y')}</p><p class='kpi-label'>First order placed</p></div>",
                         unsafe_allow_html=True)
            tl2.markdown(f"<div class='kpi-card'><p class='kpi-value' style='color:{COLOUR_LOW};'>"
                         f"{last_dt.strftime('%b %Y')}</p><p class='kpi-label'>Last order placed</p></div>",
                         unsafe_allow_html=True)
            tl3.markdown(f"<div class='kpi-card'><p class='kpi-value' style='color:{inact_col};'>"
                         f"{inactive_days} days</p><p class='kpi-label'>Since last order (as of {_data_as_of})</p></div>",
                         unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            min_inactive_pct = max(inactive_pct, 4)
            real_active_pct  = 100 - min_inactive_pct
            st.markdown(
                f"<div style='margin:4px 0 4px;font-size:12px;color:#888;'>"
                f"Customer relationship timeline &nbsp;"
                f"<span style='color:{COLOUR_LOW};font-weight:600;'>&#9632; Active ({active_pct}%)</span>"
                f"&nbsp;&nbsp;"
                f"<span style='color:{inact_col};font-weight:600;'>&#9632; Inactive ({inactive_pct}%)</span>"
                f"</div>"
                f"<div style='display:flex;height:30px;border-radius:6px;overflow:hidden;width:100%;'>"
                f"<div style='width:{real_active_pct}%;background:{COLOUR_LOW};display:flex;"
                f"align-items:center;justify-content:center;font-size:12px;color:white;font-weight:600;'>"
                f"{cb['relationship_years']:.1f} yrs active</div>"
                f"<div style='width:{min_inactive_pct}%;background:{inact_col};display:flex;"
                f"align-items:center;justify-content:center;font-size:12px;color:white;font-weight:600;'>"
                f"{inactive_days}d gap</div>"
                f"</div>",
                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if inactive_days > 365:
                st.markdown(
                    f"<div class='callout-danger'><strong>No orders in {inactive_days} days "
                    f"({inactive_days//365} year(s)).</strong> Customers who stop buying frequently "
                    f"struggle to pay outstanding balances — this is a key risk indicator.</div>",
                    unsafe_allow_html=True)
            elif inactive_days > 180:
                st.markdown(
                    f"<div class='callout-warning'><strong>No orders in {inactive_days} days.</strong> "
                    f"Consider proactive outreach to understand if there are any underlying issues.</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div class='callout-success'>Last order placed {inactive_days} days ago. "
                    f"This customer remains actively engaged.</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-header'>How this customer compares to their segment</div>",
                        unsafe_allow_html=True)
            seg_name = cust.get("cluster_name", None)
            if seg_name and seg_name in behaviour_df.get("cluster_name", pd.Series()).values:
                seg_peers = behaviour_df[behaviour_df["cluster_name"]==seg_name]
                seg_avg   = seg_peers[["total_sales_value","purchase_frequency","recency_days","unique_products"]].mean()
                comp_config = [
                    ("Total lifetime purchases", cb["total_sales_value"],  seg_avg["total_sales_value"],  "${:,.0f}", True),
                    ("Orders per month",          cb["purchase_frequency"], seg_avg["purchase_frequency"], "{:.2f}",   True),
                    ("Days since last order",     cb["recency_days"],       seg_avg["recency_days"],       "{:.0f}d",  False),
                    ("Products ordered",          cb["unique_products"],    seg_avg["unique_products"],    "{:.0f}",   True),
                ]
                rows = []
                below = []
                for label, cval, sval, fmt, higher_better in comp_config:
                    pct = (cval/sval*100) if sval else 0
                    above = pct >= 100 if higher_better else pct <= 100
                    arrow = "↑ Above average" if above else "↓ Below average"
                    if not above: below.append(label)
                    cval_s = fmt.format(cval) if not fmt.startswith("$") else "$" + f"{cval:,.0f}"
                    sval_s = fmt.format(sval) if not fmt.startswith("$") else "$" + f"{sval:,.0f}"
                    rows.append({"Metric":label,"This customer":cval_s,"Segment average":sval_s,
                                 "vs Segment":f"{arrow} ({pct:.0f}%)"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                if below:
                    bl = " and ".join([f"<strong>{m}</strong>" for m in below])
                    st.markdown(
                        f"<div class='callout-warning'>Compared to other {seg_name} customers, "
                        f"this account is underperforming on: {bl}. "
                        f"This divergence from their peer group is a behavioural risk signal.</div>",
                        unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("<div class='section-header'>Relationship summary</div>", unsafe_allow_html=True)
                profile_row("Customer since",       first_dt.strftime("%b %Y"))
                profile_row("Last order placed",    last_dt.strftime("%b %Y"))
                profile_row("Days since last order", f"{int(cb['recency_days'])} days")
                profile_row("Years as customer",    f"{cb['relationship_years']:.1f} yrs")
            with col_r:
                st.markdown("<div class='section-header'>Product purchasing</div>", unsafe_allow_html=True)
                profile_row("Unique products ordered", int(cb["unique_products"]))
                if "unique_categories" in cb.index: profile_row("Product categories", int(cb["unique_categories"]))
                if "top_category" in cb.index:      profile_row("Primary product category", cb["top_category"])
        else:
            st.info("No purchase history found for this customer.")


# PAGE 3: SEGMENTS & EXPOSURE 
elif page == "👥  Segments & Exposure":
    st.title("Customer Segments & Financial Exposure")
    st.markdown("<div class='callout-info'>Customers are grouped by purchasing behaviour. "
                "Understanding which segments carry the most risk helps the credit team prioritise collection efforts.</div>",
                unsafe_allow_html=True)

    tab_seg, tab_exp = st.tabs(["Customer Segments", "Product & Overdue Exposure"])

    with tab_seg:
        col_pca, col_table = st.columns([1.2, 1])
        with col_pca:
            st.markdown("<div class='section-header'>Customer segment map</div>", unsafe_allow_html=True)
            if "pca_x" in behaviour_df.columns and "pca_y" in behaviour_df.columns:
                beh_risk = behaviour_df.merge(risk_df[["customerid","risk_bucket","risk_score"]],
                                              on="customerid", how="left")
                nc2 = next((c for c in ["customer","customername"] if c in beh_risk.columns), None)
                hd  = {"pca_x":False,"pca_y":False,"risk_score":True,"cluster_name":False}
                if nc2: hd[nc2] = True
                fig_pca = px.scatter(beh_risk, x="pca_x", y="pca_y", color="cluster_name",
                    hover_data=hd, labels={"pca_x":"","pca_y":"","cluster_name":"Segment"},
                    color_discrete_sequence=px.colors.qualitative.Set2)
                fig_pca.update_traces(marker=dict(size=7, opacity=0.7))
                plotly_defaults(fig_pca)
                fig_pca.update_layout(height=340, xaxis=dict(showticklabels=False),
                                       yaxis=dict(showticklabels=False,showgrid=False))
                st.plotly_chart(fig_pca, use_container_width=True)
                st.caption("Each dot is a customer. Colour = segment (K-Means k=5, PCA 2D). "
                           "Customers closer together have similar buying behaviour.")
            else:
                st.info("Segment map not available.")

        with col_table:
            st.markdown("<div class='section-header'>Segment risk summary</div>", unsafe_allow_html=True)
            seg_risk = risk_df.groupby("cluster_name").agg(
                customers=("customerid","count"),
                avg_score=("risk_score","mean"),
                high_risk=("risk_bucket", lambda x: (x=="High Risk").sum()),
                outstanding=("total_outstanding_usd","sum")).reset_index()
            seg_risk["% At Risk"] = (seg_risk["high_risk"]/seg_risk["customers"]*100).round(0).astype(int)
            seg_risk["avg_score"] = seg_risk["avg_score"].round(0).astype(int)
            seg_risk["outstanding"] = seg_risk["outstanding"].apply(lambda x: f"${x/1e6:.1f}M")
            seg_risk = seg_risk.sort_values("avg_score", ascending=False)
            seg_risk.columns = ["Segment","Customers","Avg Score","High Risk","Outstanding","% At Risk"]
            st.dataframe(seg_risk[["Segment","Customers","Avg Score","Outstanding","High Risk","% At Risk"]],
                         use_container_width=True, hide_index=True)

            churned = risk_df[risk_df["cluster_name"]=="Churned / Inactive"]
            if not churned.empty:
                pct_c = len(churned)/len(risk_df)*100
                pct_o = churned["total_outstanding_usd"].sum()/risk_df["total_outstanding_usd"].sum()*100
                st.markdown(
                    f"<div class='callout-danger' style='margin-top:10px;'>Churned and inactive customers "
                    f"make up {pct_c:.0f}% of accounts but hold {pct_o:.0f}% of total outstanding.</div>",
                    unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Risk breakdown by customer segment</div>", unsafe_allow_html=True)
        st.caption("Shows how many customers in each segment are High, Medium, or Low risk. "
                   "Segments with a large High Risk bar are the credit team's priority.")
        seg_dist = risk_df.groupby(["cluster_name","risk_bucket"]).size().reset_index(name="count")
        fig_sd = px.bar(seg_dist, x="cluster_name", y="count", color="risk_bucket", barmode="group",
            color_discrete_map={"High Risk":COLOUR_HIGH,"Medium Risk":COLOUR_MED,"Low Risk":COLOUR_LOW},
            labels={"cluster_name":"","count":"Number of customers","risk_bucket":""}, text="count")
        fig_sd.update_traces(texttemplate="%{text}", textposition="outside")
        plotly_defaults(fig_sd)
        fig_sd.update_layout(height=320, yaxis=dict(range=[0, seg_dist["count"].max()*1.2]))
        st.plotly_chart(fig_sd, use_container_width=True)

        st.markdown("<div class='section-header'>Behavioural signals by risk level</div>", unsafe_allow_html=True)
        risk_beh = risk_df.merge(behaviour_df[["customerid","recency_days","total_sales_value","purchase_frequency"]],
                                  on="customerid", how="left")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            ar = risk_beh.groupby("risk_bucket")["recency_days"].mean().round(0).reset_index()
            ar.columns = ["Risk Level","Days"]
            fig_r1 = px.bar(ar, x="Risk Level", y="Days", color="Risk Level",
                color_discrete_map={"High Risk":COLOUR_HIGH,"Medium Risk":COLOUR_MED,"Low Risk":COLOUR_LOW},
                text="Days", labels={"Days":"Days","Risk Level":""}, title="Days since last order")
            fig_r1.update_traces(texttemplate="%{text:.0f}d", textposition="outside")
            plotly_defaults(fig_r1)
            fig_r1.update_layout(height=260, showlegend=False, yaxis=dict(range=[0,ar["Days"].max()*1.3]))
            st.plotly_chart(fig_r1, use_container_width=True)
            st.caption("High risk accounts stopped buying earlier on average.")

        with col_r2:
            af = risk_beh.groupby("risk_bucket")["purchase_frequency"].mean().round(2).reset_index()
            af.columns = ["Risk Level","Freq"]
            fig_r2 = px.bar(af, x="Risk Level", y="Freq", color="Risk Level",
                color_discrete_map={"High Risk":COLOUR_HIGH,"Medium Risk":COLOUR_MED,"Low Risk":COLOUR_LOW},
                text="Freq", labels={"Freq":"Orders/month","Risk Level":""}, title="Purchase frequency")
            fig_r2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            plotly_defaults(fig_r2)
            fig_r2.update_layout(height=260, showlegend=False, yaxis=dict(range=[0,af["Freq"].max()*1.3]))
            st.plotly_chart(fig_r2, use_container_width=True)
            st.caption("Low risk accounts order more consistently.")

        with col_r3:
            av = risk_beh.groupby("risk_bucket")["total_sales_value"].mean().round(0).reset_index()
            av.columns = ["Risk Level","Sales"]
            fig_r3 = px.bar(av, x="Risk Level", y="Sales", color="Risk Level",
                color_discrete_map={"High Risk":COLOUR_HIGH,"Medium Risk":COLOUR_MED,"Low Risk":COLOUR_LOW},
                text=av["Sales"].apply(lambda x: f"${x:,.0f}"),
                labels={"Sales":"Avg Sales ($)","Risk Level":""}, title="Average lifetime purchase value")
            fig_r3.update_traces(textposition="outside")
            plotly_defaults(fig_r3)
            fig_r3.update_layout(height=260, showlegend=False, yaxis=dict(range=[0,av["Sales"].max()*1.3]))
            st.plotly_chart(fig_r3, use_container_width=True)
            st.caption("Purchase value alone does not determine risk.")

    with tab_exp:
        st.markdown("<div class='section-header'>Outstanding balance by product category</div>",
                    unsafe_allow_html=True)
        if "top_category" in risk_df.columns:
            cat_df = risk_df.groupby("top_category").agg(
                outstanding_usd=("total_outstanding_usd","sum"),
                avg_risk_score=("risk_score","mean"),
                customers=("customerid","count"),
                high_risk=("risk_bucket", lambda x: (x=="High Risk").sum())).reset_index()
            cat_df = cat_df[cat_df["top_category"].ne("Unknown")]
            cat_df["avg_risk_score"] = cat_df["avg_risk_score"].round(1)
            fig_tree = px.treemap(cat_df, path=["top_category"], values="outstanding_usd",
                color="avg_risk_score",
                color_continuous_scale=["#27ae60","#f39c12","#e74c3c"], range_color=[0,100],
                hover_data={"customers":True,"high_risk":True},
                labels={"outstanding_usd":"Outstanding (USD)","avg_risk_score":"Risk Score",
                        "customers":"Accounts","high_risk":"High Risk Accounts"})
            fig_tree.update_traces(texttemplate="<b>%{label}</b><br>$%{value:,.0f}", textfont_size=13)
            fig_tree.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10,b=10,l=10,r=10), coloraxis_colorbar=dict(title="Risk Score"))
            st.plotly_chart(fig_tree, use_container_width=True)
            st.caption("Box size = outstanding balance. Colour = average risk score. Hover to see account counts.")
        else:
            st.info("Product category data not available.")

        st.markdown("<br>", unsafe_allow_html=True)
        col_od, col_pay = st.columns(2)

        with col_od:
            st.markdown("<div class='section-header'>Ageing of outstanding invoices across the portfolio</div>",
                        unsafe_allow_html=True)
            st.caption("Each account is placed in a band by its single longest unpaid invoice. "
                       "Red bands (91+ days) are the highest collection priority.")
            bins        = [0, 30, 60, 90, 180, float("inf")]
            labels_bins = ["0-30 days","31-60 days","61-90 days","91-180 days","Over 180 days"]
            risk_df["overdue_band"] = pd.cut(risk_df["max_overdue_days"], bins=bins, labels=labels_bins, right=True)
            od_dist = risk_df["overdue_band"].value_counts().sort_index().reset_index()
            od_dist.columns = ["Overdue period","Accounts"]
            od_dist["pct"]   = (od_dist["Accounts"]/od_dist["Accounts"].sum()*100).round(1)
            od_dist["label"] = od_dist.apply(lambda r: f"{r['Accounts']} accounts ({r['pct']:.0f}%)", axis=1)
            fig_od = px.bar(od_dist, x="Overdue period", y="Accounts", color="Overdue period",
                color_discrete_sequence=[COLOUR_LOW,"#8BC34A",COLOUR_MED,"#FF7043",COLOUR_HIGH],
                labels={"Overdue period":"","Accounts":"Number of accounts"}, text="label")
            fig_od.update_traces(textposition="outside")
            plotly_defaults(fig_od)
            fig_od.update_layout(height=320, showlegend=False,
                                  yaxis=dict(range=[0, od_dist["Accounts"].max()*1.3]))
            st.plotly_chart(fig_od, use_container_width=True)

        with col_pay:
            st.markdown("<div class='section-header'>Risk score: accounts with vs without unallocated payments</div>",
                        unsafe_allow_html=True)
            risk_df["has_receipt"] = risk_df["receipt_count"] > 0
            rr = risk_df.groupby("has_receipt").agg(
                avg_score=("risk_score","mean"),
                count=("customerid","count")).reset_index()
            rr["Account type"] = rr["has_receipt"].map({True:"Has unallocated payments",False:"No payments on account"})
            rr["avg_score"]    = rr["avg_score"].round(1)
            rr["label"]        = rr.apply(lambda r: f"Avg score: {r['avg_score']}  ({int(r['count'])} accounts)", axis=1)
            hs = rr[rr["has_receipt"]==True]["avg_score"].values
            ns = rr[rr["has_receipt"]==False]["avg_score"].values
            cmap = {"Has unallocated payments": COLOUR_LOW if (len(hs) and len(ns) and hs[0]<ns[0]) else COLOUR_MED,
                    "No payments on account":   COLOUR_HIGH if (len(hs) and len(ns) and ns[0]>hs[0]) else COLOUR_MED}
            fig_pay = px.bar(rr, x="Account type", y="avg_score", color="Account type",
                color_discrete_map=cmap, text="label",
                labels={"avg_score":"Average risk score (0-100)","Account type":""})
            fig_pay.update_traces(texttemplate="%{text}", textposition="outside")
            plotly_defaults(fig_pay)
            fig_pay.update_layout(height=320, showlegend=False, yaxis_range=[0,100])
            st.plotly_chart(fig_pay, use_container_width=True)
            st.caption("Unallocated payments are receipts not yet matched to specific invoices. "
                       "This chart shows whether recent payment activity correlates with lower risk across the portfolio.")
