import streamlit as st

# 1. Define your pages
# You can point to any file. 'streamlit_app.py' is your main dashboard.
dashboard = st.Page("streamlit_app.py", title="Dashboard", default=True)
spreadfinder = st.Page("spread_finder.py", title="Spread Finder")
logreview = st.Page("log_review.py", title="Log Review")

# You can add more pages here easily
# settings = st.Page("pages/settings.py", title="Settings", icon="⚙️")

# 2. Define Navigation with SECTIONS (The Dictionary)
# The keys ("Main", "Analysis") will appear as gray section headers in the sidebar
pg = st.navigation({
    "Main": [dashboard],
    "Spread Finder": [spreadfinder],
    "Log Review": [logreview],
    # "Configuration": [settings] 
})

# 3. Run the app
pg.run()
