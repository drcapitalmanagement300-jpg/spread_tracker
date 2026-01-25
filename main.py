import streamlit as st

# 1. Define your pages
# The 'default=True' here ensures Dashboard loads first
dashboard = st.Page("streamlit_app.py", title="Dashboard", default=True)
macro = st.Page("macro_dashboard.py", title="Macro Command Center") # <--- NEW PAGE
#spreadfinder = st.Page("spread_finder.py", title="Spread Finder") 
logreview = st.Page("log_review.py", title="Log Review")
cornwall = st.Page("cornwall_hunter.py", title="Cornwall Hunter") 
#backtesting = st.Page("backtesting.py", title="Options Lab")

# 2. Define Navigation
# Order here determines the sidebar order
pg = st.navigation([macro, dashboard, logreview, cornwall])

# Unhash below if you want to enable the Spread Finder or Backtesting later
#pg = st.navigation([dashboard, macro, spreadfinder, logreview, cornwall, backtesting])

# 3. Run the app
pg.run()
