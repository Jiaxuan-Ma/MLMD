
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

    colored_header(label="LETs GO ",description=" ",color_name="violet-70")


    P1 = card(
    title="DATA VISUALIZATION!",
    text=" ",
    image=  "https://tse1-mm.cn.bing.net/th/id/R-C.75ddb34c910f531307d15a3c8cfabd99?rik=UjB2rVbX41xGiQ&riu=http%3a%2f%2fpic.zsucai.com%2ffiles%2f2013%2f0830%2fxiaguang1.jpg&ehk=qLHz0WQu%2bbLSDyX04QscddrbMoX%2bm5wAR8hHIyiGvvA%3d&risl=&pid=ImgRaw&r=0")
    if P1:
        switch_page("CONTINUOUS TARGETs")
        


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# streamlit_analytics.stop_tracking(unsafe_password="test123")



