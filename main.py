import streamlit as st

# Define your pages here
dashboard = st.Page("streamlit_app.py", title="Dashboard", icon="📊")
finder = st.Page("spread_finder.py", title="Deep Dive", icon="📈")

# Create the navigation
pg = st.navigation([dashboard, finder])

# Run the selected page
pg.run()
