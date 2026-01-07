import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML-PID Parkinson's Predictor")

st.title("🧠 Parkinson's Disease Predictor")
st.subheader("Machine Learning + PID Controller Prototype")

# Sidebar inputs
st.sidebar.header("Patient Signal Input")
voice = st.sidebar.slider("Voice Tremor Level", 0.0, 1.0, 0.5)
gait = st.sidebar.slider("Gait Instability", 0.0, 1.0, 0.4)
rigidity = st.sidebar.slider("Muscle Rigidity", 0.0, 1.0, 0.3)

# Simulated ML output
ml_output = (voice + gait + rigidity) / 3

# PID controller simulation
Kp, Ki, Kd = 1.5, 0.2, 0.05
error = 0.5 - ml_output
pid_output = Kp * error

final_output = ml_output + pid_output

# Display results
st.write("### ML Model Output Probability:", round(ml_output, 3))
st.write("### After PID Correction:", round(final_output, 3))

# Plot
fig, ax = plt.subplots()
ax.bar(["ML Output", "PID Corrected"], [ml_output, final_output])
ax.set_ylabel("Parkinson's Probability")
st.pyplot(fig)

# Final decision
if final_output > 0.5:
    st.error("🟥 Parkinson's Disease Detected")
else:
    st.success("🟩 Healthy Subject")
