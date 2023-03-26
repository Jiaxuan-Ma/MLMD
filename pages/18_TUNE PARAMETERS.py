from utils import *


    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================
image = Image.open('logo.png')

st.sidebar.image(image)

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================

st.write('## TUNE PARAMETERs')
st.write('---')

# =====================================================

if st.session_state["authentication_status"]:
    col1,col2 = st.columns([1,1])
    with col1:
        P1 = card(
        title="TUNE DECISION TREE PARAMETERs!",
        text="",
        image= "https://wallpaperaccess.com/full/156340.jpg")
        if P1:
            switch_page("TUNE DECISION TREE PARAMETERs")
    with col2:
        P2 = card(
        title="TUNE RANDOM FOREST PARAMETERs!",
        text="",
        image= "https://images.alphacoders.com/112/112121.jpg")
        if P2:
            switch_page("TUNE RANDOM FOREST PARAMETERs")
    
        
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


