import streamlit as st

# 1. Define your pages
dashboard = st.Page("streamlit_app.py", title="Dashboard", default=True)
spreadfinder = st.Page("spread_finder.py", title="Spread Finder")
logreview = st.Page("log_review.py", title="Log Review")

# 2. Define Navigation
# By passing a simple LIST (brackets []) instead of a dictionary, 
# Streamlit renders a flat menu without section headers.
pg = st.navigation([dashboard, spreadfinder, logreview])

# 3. Run the app
pg.run()
