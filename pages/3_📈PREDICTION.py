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

    col1, col2 = st.columns([2,2])

    with col1:
        P1 = card(
        title="REGRESSION!",
        text="",
        image= "https://ts1.cn.mm.bing.net/th/id/R-C.12f82aa380444b1fddae6c4076edc71b?rik=MU0uAw3%2fegKZaw&riu=http%3a%2f%2fpic.zsucai.com%2ffiles%2f2013%2f0717%2fbingo7.jpg&ehk=w43Ou14tQlO6vvwZRS%2bDVUUxJu8xCA6SoyEfdoszXxA%3d&risl=&pid=ImgRaw&r=0")
        if P1:
            switch_page("REGRESSION")

    with col2:

        P2 = card(
        title="CLASSIFICATION!",
        text="",
        image="https://ts1.cn.mm.bing.net/th/id/R-C.f55aa038bb67c75d84d5445050f76239?rik=EUpLueK%2bQk6yRA&riu=http%3a%2f%2fwallpapercave.com%2fwp%2fFCDgjHU.jpg&ehk=CKI50JeZ6WVUnmhfZKi70pjBJ%2f3fg1VS34l5vg2nppY%3d&risl=&pid=ImgRaw&r=0")
        if P2:
            switch_page("CLASSIFICATION")


    col1, col2 = st.columns([2,2])
    with col1:

        P3 = card(
        title="CLUSTER!",
        text="",
        image="https://tse1-mm.cn.bing.net/th/id/OIP-C.yIRNIl3WJT-3j0rNT9lQIAHaEo?w=283&h=180&c=7&r=0&o=5&dpr=1.4&pid=1.7")
        if P3:
            switch_page("CLUSTER")


    with col2:
        P4 = card(
        title="TUNE PARAMETERs!",
        text="",
        image= "https://tse1-mm.cn.bing.net/th/id/OIP-C.URx5TJAByHUUCrx1z0U9igHaE8?pid=ImgDet&rs=1")
        if P4:
            switch_page("TUNE PARAMETERs")

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


