import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration and Setup ---

# Set wide layout for better graph visibility
st.set_page_config(layout="wide", page_title="SynchroMix AI - Presentation Dummy")

# Use Tailwind-like classes (via Markdown) for better styling where possible
# Streamlit usually handles basic theming well, but we can enhance the title.
st.markdown(
    """
    <style>
    .big-title {
        font-size: 40px;
        font-weight: 700;
        color: #FF4B4B; /* Streamlit Red/Primary Color */
        text-align: center;
        padding-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        font-weight: 300;
        text-align: center;
        color: #7f7f7f;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="big-title">🎧 SynchroMix AI: Intelligent Music Transition Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Simulating a seamless DJ set using advanced audio analysis.</p>', unsafe_allow_html=True)
st.write("---")

# --- Dummy Data Generation Functions ---

def generate_dummy_data(steps=100):
    """Generates placeholder data for music analysis charts."""
    time = np.linspace(0, 10, steps) # 10 seconds of transition
    
    # 1. BPM Alignment
    bpm_track_a = np.ones(steps) * 128.0 + np.random.randn(steps) * 0.1
    bpm_track_b = np.ones(steps) * 128.5
    bpm_track_b[20:70] = np.linspace(128.0, 128.5, 50)
    
    # 2. Spectral Overlap Score (0 to 100)
    spectral_score = np.interp(time, [0, 5, 10], [10, 85, 25]) + np.random.randn(steps) * 5
    spectral_score = np.clip(spectral_score, 0, 100)
    
    # 3. Energy Level (Track A fading out, Track B fading in)
    energy_a = 100 * np.exp(-time/4) + np.random.randn(steps) * 2
    energy_b = 100 * (1 - np.exp(-time/4)) + np.random.randn(steps) * 2
    
    # 4. Harmonic Key Prediction (Categorical, mapped to numbers)
    # 1=Am, 2=C, 3=Fm, 4=Ab
    key_a = np.ones(steps) * 2
    key_b = np.ones(steps) * 2
    
    # 5. Real-time EQ Filter (Hz vs Gain)
    freqs = np.logspace(1, 4, 100) # 10Hz to 10kHz
    gain = 5 * np.sin(freqs / 500) * np.exp(-freqs/3000) # Simulated high-pass filter curve
    
    return time, bpm_track_a, bpm_track_b, spectral_score, energy_a, energy_b, freqs, gain

TIME, BPM_A, BPM_B, SPECTRAL_SCORE, ENERGY_A, ENERGY_B, FREQS, GAIN = generate_dummy_data()

# --- Layout: Main Visualization Area (Top Row) ---

st.header("1. Core Transition Analysis: BPM and Energy Flow")
col1, col2 = st.columns(2)

with col1:
    # --- Graph 1: BPM Alignment / Tempo Curve ---
    st.subheader("Graph 1: Real-Time BPM Alignment Curve")
    st.markdown("Shows the model's gradual tempo correction (Track B) to match the outgoing track (Track A).")
    
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(TIME, BPM_A, label='Track A (Source)', color='red', linestyle='--')
    ax1.plot(TIME, BPM_B, label='Track B (Target)', color='blue')
    ax1.axvline(x=5, color='gray', linestyle=':', label='Mix Point (5s)')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("BPM")
    ax1.legend()
    ax1.set_title("Target Track Tempo Adjustment (BPM)")
    st.pyplot(fig1)

with col2:
    # --- Graph 2: Energy & Loudness Trend ---
    st.subheader("Graph 2: Perceived Energy & Loudness Trend")
    st.markdown("Ensures a smooth transfer of perceived dance-floor energy (RMS loudness/intensity).")
    
    df_energy = pd.DataFrame({
        'Time': TIME,
        'Track A Energy': ENERGY_A,
        'Track B Energy': ENERGY_B
    }).set_index('Time')
    
    st.area_chart(df_energy)
    st.markdown("""
        * **Interpretation:** Track A's energy smoothly decreases as Track B's energy increases, preventing 'dips' or 'peaks' in perceived loudness.
        * **Mix Duration:** 10s
    """)

# --- Layout: Detailed Analysis Area (Bottom Section) ---

st.header("2. Harmonic and Spectral Deep Dive")
col3, col4, col5 = st.columns(3)

with col3:
    # --- Graph 3: Harmonic/Key Compatibility Score ---
    st.subheader("Graph 3: Harmonic Compatibility Score")
    st.markdown("A proprietary score (0-100) derived from **Camelot Key** and **Tonal Balance** metrics.")
    
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.plot(TIME, SPECTRAL_SCORE, label='Compatibility', color='green', linewidth=3)
    ax3.axhspan(80, 100, color='green', alpha=0.1, label='Ideal Range')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Score (0-100)")
    ax3.legend(loc='lower left')
    ax3.set_title("Harmonic Match Confidence")
    st.pyplot(fig3)
    
with col4:
    # --- Graph 4: Dynamic EQ/Filter Simulation ---
    st.subheader("Graph 4: Real-Time EQ Filter Curve")
    st.markdown("Visualization of the dynamic 3-band EQ adjustments applied during the crossfade.")
    
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    ax4.semilogx(FREQS, GAIN, label='Applied Filter', color='purple')
    ax4.axhline(0, color='gray', linestyle='--')
    ax4.set_xlabel("Frequency (Hz) [Log Scale]")
    ax4.set_ylabel("Gain (dB)")
    ax4.set_title("Model's Live High-Pass Filter")
    st.pyplot(fig4)

with col5:
    # --- Graph 5: Transition Waveform Overlap ---
    st.subheader("Graph 5: Transition Waveform Zoom")
    st.markdown("Micro-analysis of the sample-level overlap at the exact moment of cross-fading.")
    
    # Simple sine wave simulation for waveform
    waveform_time = np.linspace(4.5, 5.5, 500)
    waveform_a = np.sin(waveform_time * 20 * np.pi) * np.exp(-(waveform_time-4.5))
    waveform_b = np.sin(waveform_time * 20 * np.pi + 0.1) * (1 - np.exp(-(waveform_time-4.5)))
    
    fig5, ax5 = plt.subplots(figsize=(6, 3))
    ax5.plot(waveform_time, waveform_a, color='red', alpha=0.5)
    ax5.plot(waveform_time, waveform_b, color='blue', alpha=0.5)
    ax5.fill_between(waveform_time, waveform_a + waveform_b, color='green', alpha=0.2)
    ax5.set_title("Waveform Phase/Amplitude Match")
    ax5.set_xticks([])
    ax5.set_yticks([])
    st.pyplot(fig5)

# --- Final Presentation Points (Sidebar/Interactive Elements) ---

st.sidebar.title("App Controls (Presentation Focus)")

# Sidebar element 1: Model Choice
st.sidebar.subheader("Select Model Version")
model_version = st.sidebar.selectbox(
    "Choose AI Transition Model:",
    ["V3.1 (Harmonic & Tempo Locked)", "V2.0 (Tempo Only)", "Manual DJ Override"]
)

# Sidebar element 2: Transition Metrics
st.sidebar.subheader("Key Transition Metrics")
st.sidebar.metric("Average Harmonic Score", "87.5%", "High")
st.sidebar.metric("Tempo Delta (Post-Sync)", "0.01 BPM", "Perfect")
st.sidebar.metric("Loudness Change (LUFS)", "+0.1 LUFS", "Ideal")

# Sidebar element 3: Interactive Demo (Placeholder)
st.sidebar.subheader("Interactive Demo")
st.sidebar.markdown(f"**Current Transition:** Track A (128.0 BPM, C Minor) → Track B (128.0 BPM, G Minor)")
st.sidebar.slider("Simulate Crossfader Position", 0, 100, 50, help="This slider would show the mix in real-time.")
if st.sidebar.button("Simulate New Transition"):
    st.balloons()
    st.sidebar.info("Transition Model Re-calculating Parameters...")
    # In a real app, this would trigger the model and refresh the graphs

# --- Expander for detailed explanation (Graph 6) ---
st.write("---")
with st.expander("Click to view: Technical Deep Dive (Graph 6: Beat Grid Drift)"):
    st.subheader("Graph 6: Beat Grid Drift and Phase Correction")
    st.markdown("This visualization shows the minute phase adjustments (in milliseconds) the model applies to maintain a perfect locked-in rhythm during the transition.")
    
    # Dummy data for Beat Grid Drift
    time_drift = np.linspace(0, 10, 100)
    drift_ms = 10 * np.sin(time_drift * 5) * np.exp(-time_drift/5) # Starts high, dampens
    
    fig6, ax6 = plt.subplots(figsize=(12, 4))
    ax6.plot(time_drift, drift_ms, label='Phase Correction', color='orange')
    ax6.axhline(0, color='red', linestyle='--', label='Ideal Phase')
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Phase Correction (ms)")
    ax6.set_title("Model's Phase Correction Magnitude")
    ax6.legend()
    st.pyplot(fig6)

st.write("This structure provides a clear, interactive, and visually rich interface suitable for explaining the core components of your music transition model.")
