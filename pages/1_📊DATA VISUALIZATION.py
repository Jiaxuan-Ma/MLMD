
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
        image=  "https://ts1.cn.mm.bing.net/th/id/R-C.58d20f1fd4c2c103c8bd327e71c519fa?rik=w5vhuBIv8msleg&riu=http%3a%2f%2fimg.pconline.com.cn%2fimages%2fupload%2fupc%2ftx%2fwallpaper%2f1209%2f14%2fc2%2f13855364_1347613545563.jpg&ehk=54bnqi6V6arP3ZdfR1nbNPOqhKOtDzpI08WFudOrhQw%3d&risl=&pid=ImgRaw&r=0")
        if P1:
            switch_page("Profiling")
    with col2:
        P2 = card(
        title="DATA VISUALIZATION!",
        text=" ",
        image=  "https://ts1.cn.mm.bing.net/th/id/R-C.3e42ad7c7744a7a612bfbaa480405611?rik=eX46Dun%2fFF1aDg&riu=http%3a%2f%2fpic.bizhi360.com%2fbbpic%2f25%2f6125.jpg&ehk=ukEQFHgNgClnZxjkosi%2bP3OoaZicxvcitCFcgJISxww%3d&risl=&pid=ImgRaw&r=0")
        if P2:
            switch_page("CONTINUOUS TARGETs")


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# streamlit_analytics.stop_tracking(unsafe_password="test123")



