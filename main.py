import streamlit as st

# 1. Define your pages
# You can point to any file. 'streamlit_app.py' is your main dashboard.
dashboard = st.Page("streamlit_app.py", title="Dashboard", icon="📊", default=True)
analytics = st.Page("spread_finder.py", title="Spread Finder", icon="📈")

# You can add more pages here easily
# settings = st.Page("pages/settings.py", title="Settings", icon="⚙️")

# 2. Define Navigation with SECTIONS (The Dictionary)
# The keys ("Main", "Analysis") will appear as gray section headers in the sidebar
pg = st.navigation({
    "Main": [dashboard],
    "Analysis": [analytics],
    # "Configuration": [settings] 
})

# 3. Run the app
pg.run()
