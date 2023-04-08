
from utils import *

# streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================


st.write('## DATA VISUALIZATION')
st.write('---')

# =====================================================
if st.session_state["authentication_status"]:

    colored_header(label="Lets Go ",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])
    with col1:
        P1 = card(
        title="DATA PROFILING!",
        text=" ",
        image=  "https://th.bing.com/th/id/OIP.ddv1QrOKa2RNCyZDdqPUsgHaE8?pid=ImgDet&w=976&h=651&rs=1")
        if P1:
            switch_page("Profiling")
    with col2:
        P2 = card(
        title="DATA VISUALIZATION!",
        text=" ",
        image=  "https://th.bing.com/th/id/OIP.mGEWhKSTMV7hY_150FFT3wHaE7?pid=ImgDet&w=720&h=479&rs=1")
        if P2:
            switch_page("Visualization")


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# streamlit_analytics.stop_tracking(unsafe_password="test123")



