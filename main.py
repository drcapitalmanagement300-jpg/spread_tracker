import streamlit as st

# 1. Define your pages
# The 'default=True' here ensures Dashboard loads first
dashboard = st.Page("streamlit_app.py", title="Dashboard", default=True) 
#spreadfinder = st.Page("spread_finder.py", title="Spread Finder")
logreview = st.Page("log_review.py", title="Log Review")
cornwall = st.Page("cornwall_hunter.py", title="Cornwall Hunter") 
#backtesting = st.Page("backtesting.py", title="Options Lab")

# 2. Define Navigation
# Order here determines the sidebar order
pg = st.navigation([dashboard, logreview, cornwall])

# Unhash below to enable backtesting.
#pg = st.navigation([spreadfinder, dashboard, logreview, cornwall, backtesting])

# 3. Run the app
pg.run()
