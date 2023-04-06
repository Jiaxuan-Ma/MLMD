from utils import *


    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================

st.write('## PREDICTION')
st.write('---')

# =====================================================

if st.session_state["authentication_status"]:

    colored_header(label="LETs GO",description=" ",color_name="violet-70")

    P1 = card(
    title="BGOLEARN!",
    text="",
    image="")
    if P1:
        switch_page("BGOLEARN")
    

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


