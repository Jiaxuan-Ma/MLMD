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

    colored_header(label="Lets Go",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])

    with col1:
        P1 = card(
        title="REGRESSION!",
        text="",
        image= "https://www.ktchost.com/blog/wp-content/uploads/2016/11/Solution1-KTCHost-678x381.jpg")
        if P1:
            switch_page("REGRESSION")

    with col2:

        P2 = card(
        title="CLASSIFICATION!",
        text="",
        image="https://www.bernhard-kniepkamp.de/wp-content/uploads/2016/11/coaching_berlin-1024x576.jpg")
        if P2:
            switch_page("CLASSIFICATION")

    P3 = card(
    title="CLUSTER!",
    text="",
    image="https://wallpaperset.com/w/full/7/3/3/420575.jpg")
    if P3:
        switch_page("CLUSTER")


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


