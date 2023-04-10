from utils import *


    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

st.write('## PREDICTION')
st.write('---')

# =====================================================

if st.session_state["authentication_status"]:

    colored_header(label="Lets Go",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])

    with col1:
        P1 = card(
        title="REGRESSION!",
        text="",
        image= "https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P1:
            switch_page("REGRESSION")

    with col2:

        P2 = card(
        title="CLASSIFICATION!",
        text="",
        image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P2:
            switch_page("CLASSIFICATION")

    P3 = card(
    title="CLUSTER!",
    text="",
    image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
    if P3:
        switch_page("CLUSTER")


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


