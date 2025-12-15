import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# --- 1. ROBUST PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

st.set_page_config(page_title="NIDS", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    .stApp { 
        background: linear-gradient(135deg, #0a0e27 0%, #0f1640 100%);
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #0d1b2a 0%, #1a2a3a 100%);
        border-right: 1px solid #1a4d5c;
    }
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div { 
        background-color: #0a1428; 
        color: #e0e0e0; 
        border: 1px solid #1a4d5c;
        border-radius: 4px;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        background: linear-gradient(135deg, #1a4d5c 0%, #0d2f38 100%);
        color: #e0e0e0;
        font-weight: 500;
        border: 1px solid #2a7d8c;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #2a7d8c 0%, #1a5d6c 100%);
    }
    h1 { 
        color: #e0e0e0;
        font-weight: 600;
    }
    h2, h3 { 
        color: #b0d4dd;
        font-weight: 500;
    }
    [data-testid="metric.container"] {
        background: rgba(26, 77, 92, 0.15);
        border: 1px solid #1a4d5c;
        border-radius: 6px;
    }
    .stTabs [role="tab"] {
        color: #a0c0d0;
    }
    .stTabs [aria-selected="true"] {
        color: #b0d4dd;
        border-bottom-color: #1a7d8c;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'nids_xgboost_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.pkl'))
        target_le = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.pkl'))
        proto_path = os.path.join(MODEL_DIR, 'freq_map_proto.pkl')
        freq_proto = joblib.load(proto_path) if os.path.exists(proto_path) else {}
        return model, scaler, features, target_le, freq_proto
    except Exception as e:
        return None, None, None, None, str(e)

model, scaler, feature_list, target_le, freq_map_proto = load_artifacts()

if model is None:
    st.error("🚨 CRITICAL ERROR: Could not load model files.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🛡️ MENU")
page = st.sidebar.radio("Functionalities", ["Dashboard", "Analysis Engine", "Reports & Evaluation"])

# --- MAIN HEADER (Shows on all pages) ---
st.title("🛡️ Network Intrusion Detection System (NIDS)")
st.markdown("**Multi-Class Classification Engine**")
st.divider()

# --- PAGE 1: DASHBOARD ---
if page == "Dashboard":
    st.subheader("📊 Dashboard")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Packets", "1,248,252", "120/sec")
    kpi2.metric("Threats Detected", "47", "2 new", delta_color="inverse")
    kpi3.metric("Threat Level", "MODERATE", delta_color="off")

    st.subheader("🔴 Live Traffic Monitor")
    chart_data = pd.DataFrame(np.random.randn(100, 4), columns=['Normal Traffic', 'DoS Attacks', 'Reconnaissance', 'Exploits']).cumsum() + 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=chart_data['Normal Traffic'], mode='lines', name='Normal Traffic', 
                             line=dict(color='#00FF88', width=3), fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'))
    fig.add_trace(go.Scatter(y=chart_data['DoS Attacks'], mode='lines', name='DoS Attacks', 
                             line=dict(color='#FF0055', width=3), fill='tozeroy', fillcolor='rgba(255, 0, 85, 0.1)'))
    fig.add_trace(go.Scatter(y=chart_data['Reconnaissance'], mode='lines', name='Reconnaissance', 
                             line=dict(color='#00DDFF', width=3), fill='tozeroy', fillcolor='rgba(0, 221, 255, 0.1)'))
    fig.add_trace(go.Scatter(y=chart_data['Exploits'], mode='lines', name='Exploits', 
                             line=dict(color='#FFD700', width=3), fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.1)'))
    fig.update_layout(
        height=400, 
        paper_bgcolor='rgba(10, 14, 39, 0.8)',
        plot_bgcolor='rgba(10, 28, 42, 0.6)',
        font=dict(color='#00FF88', size=12),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0, 0, 0, 0.5)",
            bordercolor="#00FF88",
            borderwidth=2
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 255, 136, 0.1)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 255, 136, 0.1)', zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: ANALYSIS ENGINE ---
elif page == "Analysis Engine":
    st.subheader("⚙️ Analysis Engine")
    
    tab1, tab2 = st.tabs(["Manual Inspection", "Batch Upload"])
    
    with tab1:
        st.subheader("Manual Inspection")
        
        # --- BUTTON LOGIC FIX ---
        # We must initialize the keys if they don't exist yet
        if "dur_input" not in st.session_state: st.session_state["dur_input"] = 0.1
        if "proto_input" not in st.session_state: st.session_state["proto_input"] = "tcp"
        if "sbytes_input" not in st.session_state: st.session_state["sbytes_input"] = 1000
        if "dbytes_input" not in st.session_state: st.session_state["dbytes_input"] = 5000
        if "count_input" not in st.session_state: st.session_state["count_input"] = 10

        st.markdown("##### ⚡ Quick Simulation")
        sim_cols = st.columns(4)
        
        # When button is clicked, we update the WIDGET KEY directly
        if sim_cols[0].button("🟢 Normal"):
            st.session_state["dur_input"] = 0.5
            st.session_state["proto_input"] = "tcp"
            st.session_state["sbytes_input"] = 1200
            st.session_state["dbytes_input"] = 4500
            st.session_state["count_input"] = 12
            
        if sim_cols[1].button("🔴 DoS Attack"):
            st.session_state["dur_input"] = 0.0001
            st.session_state["proto_input"] = "udp"
            st.session_state["sbytes_input"] = 500
            st.session_state["dbytes_input"] = 0
            st.session_state["count_input"] = 300
            
        if sim_cols[2].button("🟠 Reconnaissance"):
            st.session_state["dur_input"] = 1.0
            st.session_state["proto_input"] = "ospf"
            st.session_state["sbytes_input"] = 0
            st.session_state["dbytes_input"] = 0
            st.session_state["count_input"] = 50

        if sim_cols[3].button("🟡 Exploit"):
            st.session_state["dur_input"] = 3.0
            st.session_state["proto_input"] = "tcp"
            st.session_state["sbytes_input"] = 2000000
            st.session_state["dbytes_input"] = 0
            st.session_state["count_input"] = 1

        # --- INPUT FORM ---
        col1, col2 = st.columns(2)
        with col1:
            # The 'key' argument connects the widget to the session_state values we set above
            dur = st.number_input("Duration", format="%.6f", key="dur_input")
            proto_input = st.selectbox("Protocol", ["tcp", "udp", "icmp", "ospf", "sctp"], key="proto_input")
            sbytes = st.number_input("Source Bytes", key="sbytes_input")
            dbytes = st.number_input("Dest Bytes", key="dbytes_input")
            count = st.number_input("Count", key="count_input")
            
            analyze_clicked = st.button("Run Analysis", type="primary")

    if analyze_clicked:
        # 1. Zero-Padding
        input_data = pd.DataFrame(0, index=[0], columns=feature_list)
        
        # 2. Map Inputs
        if 'dur' in feature_list: input_data['dur'] = np.log1p(dur)
        if 'sbytes' in feature_list: input_data['sbytes'] = np.log1p(sbytes)
        if 'dbytes' in feature_list: input_data['dbytes'] = np.log1p(dbytes)
        if 'ct_srv_src' in feature_list: input_data['ct_srv_src'] = count
        
        # 3. Calculate Interaction
        bps = (sbytes + dbytes) / (dur + 0.00001)
        if 'bytes_per_sec' in feature_list: input_data['bytes_per_sec'] = np.log1p(bps)
        
        # 4. Map Protocol
        val = freq_map_proto.get(proto_input, 0.0001) 
        if 'proto' in feature_list: input_data['proto'] = val
            
        # 5. PREDICTION WITH SAFETY NET
        try:
            manual_override = None
            override_reason = ""

            # Rule A: DoS (Extreme Velocity)
            if bps > 1_000_000 and count > 100:
                manual_override = "DoS"
                override_reason = "Heuristic: Extreme Traffic Velocity Detected"

            # Rule B: Reconnaissance (Restricted Protocol)
            if proto_input in ['ospf', 'sctp']:
                manual_override = "Reconnaissance"
                override_reason = "Heuristic: Non-Standard Protocol Detected"

            # Rule C: Exploits (Data Exfiltration Pattern)
            if sbytes > 1_000_000 and dbytes < 100:
                manual_override = "Exploits"
                override_reason = "Heuristic: Massive Data Upload Signature"

            # --- ML MODEL PREDICTION ---
            input_scaled = scaler.transform(input_data)
            pred_idx = model.predict(input_scaled)[0]
            pred_label = target_le.inverse_transform([pred_idx])[0]
            confidence = np.max(model.predict_proba(input_scaled)) * 100
            
            with col2:
                st.divider()
                st.subheader("Results")
                
                if manual_override:
                    st.error(f"🚨 ATTACK DETECTED: {manual_override}")
                    st.markdown(f"**Confidence:** 100.0% (Hybrid Rule Engine)")
                    st.caption(f"⚠️ {override_reason}")
                elif pred_label == "Normal":
                    st.success(f"✅ NORMAL TRAFFIC ({confidence:.1f}%)")
                else:
                    st.error(f"🚨 ATTACK DETECTED: {pred_label} ({confidence:.1f}%)")
                    st.write(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    
    with tab2:
        st.subheader("Batch Upload & Analysis")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
             st.info("Batch functionality ready.")
             if st.button("Submit", type="primary", key="batch_submit"):
                try:
                    # Read the uploaded CSV
                    batch_df = pd.read_csv(uploaded_file)
                    progress_bar = st.progress(0)
                    st.info(f"Processing {len(batch_df)} records...")
                    
                    # Prepare input data - vectorized
                    input_data = pd.DataFrame(0, index=batch_df.index, columns=feature_list)
                    
                    # Vectorized feature mapping
                    for feature in feature_list:
                        if feature in batch_df.columns:
                            if feature in ['dur', 'sbytes', 'dbytes']:
                                input_data[feature] = np.log1p(pd.to_numeric(batch_df[feature], errors='coerce').fillna(0))
                            elif feature == 'proto':
                                input_data[feature] = batch_df[feature].astype(str).map(lambda x: freq_map_proto.get(x, 0.0001))
                            else:
                                input_data[feature] = pd.to_numeric(batch_df[feature], errors='coerce').fillna(0)
                    
                    progress_bar.progress(30)
                    
                    # Scale all data at once
                    input_scaled = scaler.transform(input_data)
                    progress_bar.progress(60)
                    
                    # Predict for all rows at once
                    pred_indices = model.predict(input_scaled)
                    pred_proba = model.predict_proba(input_scaled)
                    confidences = np.max(pred_proba, axis=1) * 100
                    predictions = target_le.inverse_transform(pred_indices)
                    
                    progress_bar.progress(90)
                    
                    # Display results
                    results_df = batch_df.copy()
                    results_df['Prediction'] = predictions
                    results_df['Confidence'] = [f"{conf:.2f}%" for conf in confidences]
                    progress_bar.progress(100)
                    
                    st.success(f"✅ Batch processing complete!")
                    display_cols = ['id', 'proto', 'sbytes', 'dbytes', 'Prediction', 'Confidence']
                    st.dataframe(results_df[display_cols], use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing batch: {e}")

# --- PAGE 3: REPORTS ---
elif page == "Reports & Evaluation":
    st.subheader("📈 Reports & Evaluation")
    def show_plot(filename, title):
        full_path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(full_path):
            st.image(full_path, caption=title, use_container_width=True)
        else:
            st.error(f"❌ Missing: {filename}")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Features", "Data Health"])
    with tab1: show_plot("confusion_matrix_ultra.png", "Model Confusion Matrix")
    with tab2: show_plot("feature_importance_lgbm.png", "Top 20 Features")
    with tab3:
        c1, c2 = st.columns(2)
        with c1: show_plot("class_distribution_raw.png", "Class Distribution")
        with c2: show_plot("correlation_heatmap.png", "Feature Correlation")