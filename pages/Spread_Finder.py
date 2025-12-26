import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
# This must be the first Streamlit command on the new page
st.set_page_config(page_title="Analytics", layout="wide")

# ---------------- Header ----------------
st.title("Analytics Dashboard")
st.markdown("---")

# ---------------- Sample Content ----------------
st.write("This is a new page. You can add distinct charts and logic here without cluttering your main dashboard.")

# Example: Reusing your dark mode chart style
data = pd.DataFrame({
    'Category': ['A', 'B', 'C'],
    'Values': [10, 20, 15]
})

fig, ax = plt.subplots(figsize=(8, 3))

# Dark Mode Chart Styling
bg_color = '#0E1117'
text_color = '#FAFAFA'
ax.set_facecolor(bg_color)
fig.patch.set_facecolor(bg_color)

ax.bar(data['Category'], data['Values'], color='#00C853')

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(text_color)
ax.spines['left'].set_color(text_color)
ax.tick_params(axis='x', colors=text_color)
ax.tick_params(axis='y', colors=text_color)
ax.set_title("Sample Analytics Chart", color=text_color)

st.pyplot(fig)
