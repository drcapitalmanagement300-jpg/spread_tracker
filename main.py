import streamlit as st

# 1. Define your pages
# The 'default=True' here ensures Dashboard loads first
dashboard = st.Page("streamlit_app.py", title="Dashboard", default=True) 
spreadfinder = st.Page("spread_finder.py", title="Spread Finder")
logreview = st.Page("log_review.py", title="Log Review")
backtesting = st.Page("backtesting.py", title="Options Lab")

# 2. Define Navigation
# Order here determines the sidebar order
pg = st.navigation([spreadfinder, dashboard, logreview, backtesting])

# 3. Run the app
pg.run()
