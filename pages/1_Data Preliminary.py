
from utils import *

# streamlit_analytics.start_tracking()

st.set_page_config(
    page_title="MLMD",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
    }
)


if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None




# ======================================================



st.write('## DATA PRELIMINARY')
st.write('---')

# =====================================================
if st.session_state["authentication_status"]:

    colored_header(label="Lets Go ",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])
    with col1:
        P1 = card(
        title="DATA PROFILING!",
        text=" ",
        image=  "https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P1:
            switch_page("Profiling")
    with col2:
        P2 = card(
        title="DATA VISUALIZATION!",
        text=" ",
        image=  "https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P2:
            switch_page("Visualization")


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# streamlit_analytics.stop_tracking(unsafe_password="test123")


