"""Combined entry point: both VOI apps in one, switched from the sidebar.

Run with:   streamlit run voi_app.py

"Main VOI App" is the original binary-decision / Naive-Bayes workflow in app.py.
"Sequential Decisions" is the Expected-Information-Gain drilling planner in eig_app.py.
Each script is still runnable on its own; this file only chooses which one to execute,
so neither had to be restructured to live here.
"""
import runpy

import streamlit as st

MAIN = "Main VOI App"
SEQUENTIAL = "Sequential Decisions"
SCRIPTS = {MAIN: "app.py", SEQUENTIAL: "eig_app.py"}
# The original app was laid out for the narrow default; the planner needs the width.
LAYOUTS = {MAIN: "centered", SEQUENTIAL: "wide"}

# set_page_config has to run before any other Streamlit call, yet the layout depends on
# which app is selected.  A widget cannot be created this early, but its value is already
# in session state by the time the script reruns after a click, so read it from there.
selected = st.session_state.get("app_mode", MAIN)
if selected not in SCRIPTS:
    selected = MAIN
st.set_page_config(page_title="Value of Information", layout=LAYOUTS[selected])

with st.sidebar:
    st.radio("App", list(SCRIPTS), key="app_mode",
             help="Switch between the two workflows.")
    st.divider()

# run_name is deliberately not "__main__": each script guards its own set_page_config on
# that name so it can still be launched directly without clashing with the call above.
runpy.run_path(SCRIPTS[selected], run_name="__page__")
