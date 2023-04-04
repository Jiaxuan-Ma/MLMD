
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
    image=  "https://ts1.cn.mm.bing.net/th/id/R-C.3e42ad7c7744a7a612bfbaa480405611?rik=eX46Dun%2fFF1aDg&riu=http%3a%2f%2fpic.bizhi360.com%2fbbpic%2f25%2f6125.jpg&ehk=ukEQFHgNgClnZxjkosi%2bP3OoaZicxvcitCFcgJISxww%3d&risl=&pid=ImgRaw&r=0")
    if P1:
        switch_page("CONTINUOUS TARGETs")
        


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# streamlit_analytics.stop_tracking(unsafe_password="test123")



