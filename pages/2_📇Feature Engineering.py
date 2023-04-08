from utils import *


    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
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
        image= "https://th.bing.com/th/id/OIP.pQ-ojbIYIahpk4O4qZ8B6AHaFK?pid=ImgDet&w=600&h=418&rs=1")
        if P1:
            switch_page("DROP MISSING FEATUREs")

    with col2:

        P2 = card(
        title="FILL MISSING FEATUREs!",
        text="",
        image="https://www.stratx-exl.com/hubfs/Invest%20in%20Innovation%20Training%20.jpg#keepProtocol")
        if P2:
            switch_page("FILL MISSING FEATUREs")

    colored_header(label="Drop Nunique Features",description=" ",color_name="violet-70")
    P3= card(
    title="DROP NUNIQUE FEATUREs!",
    text="",
    image="https://th.bing.com/th/id/R.ad3496b85647db847daeeb5c791e405a?rik=eR3JmekNKe1%2beg&riu=http%3a%2f%2ftangibleinteriors.com%2fblog%2fwp-content%2fuploads%2f2015%2f08%2fidea-1.jpg&ehk=5LyLsfnVPNEd3OyECYea75KvYWXevfifC5w5KpY5Tgk%3d&risl=&pid=ImgRaw&r=0")
    if P3:
        switch_page("DROP NUNIQUE FEATUREs")
    
    colored_header(label="Correlation of Features vs Targets",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])

    with col1:

        P4 = card(
        title="DROP LOW CORRELATION FEATUREs vs TARGET!",
        text="",
        image="https://images.squarespace-cdn.com/content/v1/602574da8aa63d3ab3e7f9ca/1613132653122-AXQK07WIEXIMIMWQ02BF/Lightbulb.jpg?format=500w")
        if P4:
            switch_page("DROP LOW CORRELATION FEATUREs vs TARGET")
        
    with col2:

        P5 = card(
        title="DROP COLLINEAR FEATUREs!",
        text="",
        image="https://marketingweek.imgix.net/content/uploads/2017/08/08172435/Succession-planning-2.jpg?auto=compress,format&q=60&w=736&h=455")
        if P5:
            switch_page("DROP COLLINEAR FEATUREs")


    colored_header(label="One-hot Encoding Features",description=" ",color_name="violet-70")
    
    col1, col2 = st.columns([2,2])
    
    P6 = card(
    title="ONE-HOT ENCODING FEATUREs!",
    text="",
    image="https://www.nasdaq.com/sites/acquia.prod/files/styles/1370x700/public/etf_trends/Keep-Your-Qualitys-Momentum-Introducing-the-Virtus-JOET-ETF-440x250.jpg?1680874597890561479")
    if P6:
        switch_page("ONE-HOT ENCODING FEATUREs")
        
    colored_header(label="Features Importance",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])
   
    P7 = card(
    title="FEATUREs IMPORTANCE!",
    text="",
    image="https://th.bing.com/th/id/R.3f4bfc341cebfab604f44045f2e3032f?rik=XmJ9nimYgKT2Kg&riu=http%3a%2f%2fwww.qsan-usa.com%2fuploads%2f7%2f8%2f0%2f3%2f78037614%2fistock-491490296_orig.jpg&ehk=5x%2bwRIgnZsAbSpB%2fKrIIn9qPCsf6QwSMaL1g3GX0taE%3d&risl=&pid=ImgRaw&r=0")
    if P7:
        switch_page("FEATUREs IMPORTANCE")
    
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


