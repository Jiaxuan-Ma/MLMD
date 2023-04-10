from utils import *


    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

st.write('## FEATURE ENGINEERING')
st.write('---')

# =====================================================

if st.session_state["authentication_status"]:

    colored_header(label="Missing Features",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])

    with col1:
        P1 = card(
        title="DROP MISSING FEATUREs!",
        text="",
        image= "https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P1:
            switch_page("DROP MISSING FEATUREs")

    with col2:

        P2 = card(
        title="FILL MISSING FEATUREs!",
        text="",
        image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P2:
            switch_page("FILL MISSING FEATUREs")

    colored_header(label="Drop Nunique Features",description=" ",color_name="violet-70")
    P3= card(
    title="DROP NUNIQUE FEATUREs!",
    text="",
    image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
    if P3:
        switch_page("DROP NUNIQUE FEATUREs")
    
    colored_header(label="Correlation of Features vs Targets",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])

    with col1:

        P4 = card(
        title="DROP LOW CORRELATION FEATUREs vs TARGET!",
        text="",
        image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P4:
            switch_page("DROP LOW CORRELATION FEATUREs vs TARGET")
        
    with col2:

        P5 = card(
        title="DROP COLLINEAR FEATUREs!",
        text="",
        image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P5:
            switch_page("DROP COLLINEAR FEATUREs")


    colored_header(label="One-hot Encoding Features",description=" ",color_name="violet-70")
    
    col1, col2 = st.columns([2,2])
    
    P6 = card(
    title="ONE-HOT ENCODING FEATUREs!",
    text="",
    image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
    if P6:
        switch_page("ONE-HOT ENCODING FEATUREs")
        
    colored_header(label="Features Importance",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])
   
    P7 = card(
    title="FEATUREs IMPORTANCE!",
    text="",
    image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
    if P7:
        switch_page("FEATUREs IMPORTANCE")
    
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


