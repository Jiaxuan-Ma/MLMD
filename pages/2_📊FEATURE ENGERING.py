from utils import *


    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================

st.write('## FEATURE ENGINEER')
st.write('---')

# =====================================================

if st.session_state["authentication_status"]:

    colored_header(label="MISSING FEATUREs",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])

    with col1:
        P1 = card(
        title="DROP MISSING FEATUREs!",
        text="",
        image= "https://ts1.cn.mm.bing.net/th/id/R-C.12f82aa380444b1fddae6c4076edc71b?rik=MU0uAw3%2fegKZaw&riu=http%3a%2f%2fpic.zsucai.com%2ffiles%2f2013%2f0717%2fbingo7.jpg&ehk=w43Ou14tQlO6vvwZRS%2bDVUUxJu8xCA6SoyEfdoszXxA%3d&risl=&pid=ImgRaw&r=0")
        if P1:
            switch_page("DROP MISSING FEATUREs")

    with col2:

        P2 = card(
        title="FILL MISSING FEATUREs!",
        text="",
        image="https://ts1.cn.mm.bing.net/th/id/R-C.f55aa038bb67c75d84d5445050f76239?rik=EUpLueK%2bQk6yRA&riu=http%3a%2f%2fwallpapercave.com%2fwp%2fFCDgjHU.jpg&ehk=CKI50JeZ6WVUnmhfZKi70pjBJ%2f3fg1VS34l5vg2nppY%3d&risl=&pid=ImgRaw&r=0")
        if P2:
            switch_page("FILL MISSING FEATUREs")

    colored_header(label="NUNIQUE FEATUREs",description=" ",color_name="violet-70")
    P3= card(
    title="DROP NUNIQUE FEATUREs!",
    text="",
    image="https://desk-fd.zol-img.com.cn/t_s960x600c5/g5/M00/02/05/ChMkJ1bKyLyICgL5AAjMbLExdzkAALIGQDsAlMACMyE427.jpg")
    if P3:
        switch_page("DROP NUNIQUE FEATUREs")
    
    colored_header(label="CORRELATION FEATUREs vs TARGET",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])

    with col1:

        P4 = card(
        title="DROP LOW CORRELATION FEATUREs vs TARGET!",
        text="",
        image="https://ts1.cn.mm.bing.net/th/id/R-C.40234919a68b35bd5b8043294e7944fd?rik=%2fRxWjy0nBdtkAQ&riu=http%3a%2f%2fpic.qqbizhi.com%2fallimg%2fbbpic%2f29%2f1229_5.jpg&ehk=B%2b3JbGRA3Xq9iIyjNBr0YCvzTMGFxwLTh4ksQvagBRY%3d&risl=&pid=ImgRaw&r=0")
        if P4:
            switch_page("DROP LOW CORRELATION FEATUREs vs TARGET")
        
    with col2:

        P5 = card(
        title="DROP COLLINEAR FEATUREs!",
        text="",
        image="https://ts1.cn.mm.bing.net/th/id/R-C.8efd83545acb3ba7137b2b5450b4ea04?rik=7teH3W0KXFmD1g&riu=http%3a%2f%2fpic.zsucai.com%2ffiles%2f2013%2f0717%2fbingo5.jpg&ehk=FG6ysy4IlDTZcJBECXlDb7yrT5cApwp0Gdo5akzrSQs%3d&risl=&pid=ImgRaw&r=0")
        if P5:
            switch_page("DROP COLLINEAR FEATUREs")


    colored_header(label="ENCODING FEATUREs",description=" ",color_name="violet-70")
    
    col1, col2 = st.columns([2,2])
    
    P6 = card(
    title="ONE-HOT ENCODING FEATUREs!",
    text="",
    image="https://ts1.cn.mm.bing.net/th/id/R-C.5ca538274811bd9853f53b46af249365?rik=%2fGw9z2%2frD9R%2bJA&riu=http%3a%2f%2fup.deskcity.org%2fpic_source%2f5c%2fa5%2f38%2f5ca538274811bd9853f53b46af249365.jpg&ehk=vUjvURB8tE4gEGeveS8aQ6Ve6MydUras1UVUElYVqPo%3d&risl=&pid=ImgRaw&r=0")
    if P6:
        switch_page("ONE-HOT ENCODING FEATUREs")
        
    colored_header(label="FEATUREs IMPORTANCE",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])
   
    P7 = card(
    title="FEATUREs IMPORTANCE!",
    text="",
    image="https://ts1.cn.mm.bing.net/th/id/R-C.93e546731d5bc64544905c1e58d6e546?rik=DJf0wHkmD9sG8A&pid=ImgRaw&r=0")
    if P7:
        switch_page("FEATUREs IMPORTANCE")
    
    colored_header(label="DATA ENHANCEMENT",description=" ",color_name="violet-70")
    
    P8 = card(
    title="DATA ENHANCEMENT!",
    text="",
    image="https://ts1.cn.mm.bing.net/th/id/R-C.c8844d225dd6253fb78f4acd4fd95020?rik=BA0CggiG0Im1gw&riu=http%3a%2f%2fpic.221600.cn%2fforum%2f201408%2f15%2f002316cigtttxa8gcjbjki.jpg&ehk=qImHVhtZQI%2bhaRNaqmm08TykmhQ5hl6bqoUWhiknTgg%3d&risl=&pid=ImgRaw&r=0")
    if P8:
        switch_page("DATA ENHANCEMENT")

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


