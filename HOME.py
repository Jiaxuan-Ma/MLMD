'''
Runs the streamlit app

Call this file in the terminal via `streamlit run app.py`
'''
from utils import *


if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'logout' not in st.session_state:
    st.session_state['logout'] = None


st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)
"""
# EASY TO MATTER DESIGN
"""
st.markdown("<br>", unsafe_allow_html=True)
"""
This is a data miner visual platform.
"""

# streamlit_analytics.start_tracking()

name, authentication_status, username = authenticator.login('Login', 'main')


# ======================================================


with st.sidebar:

    badge(type="buymeacoffee", name="jiaxuanmasw")

# ======================================================

if username == "mjx@shu.edu.cn":
    # register
    colored_header(label="Register user",description=" ",color_name="violet-70")
    with st.expander('REGISTER USER'):
        try:
            if authenticator.register_user('Register user', preauthorization=False):
                st.success('User registered successfully')
        except Exception as e:
            st.error(e)
 
    with st.expander('SEND NEW PASSWORD'):
        try:
            username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Send new password')
            msg_from = 'EasyToMatterDesign@163.com'
            passwd = 'AXOFYNUMFQCLXMVM'
            receiver = email_forgot_password
            to = [receiver]
            # email content
            msg = MIMEMultipart()

            content = ('''Dear member:

                    We are writing to inform you that the password for your Easy to Matter Design account has been sent. Please check your email inbox for the password %s.
                    

                    If you have any questions or concerns, please do not hesitate to contact our member support team. We are always available to assist you.

                    Thank you for choosing Easy to Matter Design.

                    Sincerely,
                    The Easy to Matter Design Team''' %(random_password))

            msg.attach(MIMEText(content,'plain','utf-8'))
            
            # email theme
            theme = 'Easy to Matter Design Register Successfully'
            msg['Subject'] = theme
            msg['From'] = msg_from

            if username_forgot_pw:
                s = smtplib.SMTP_SSL("smtp.163.com", 465)
                s.login(msg_from, passwd)
                s.sendmail(msg_from,to,msg.as_string())
                st.success('New password sent securely')
                # Random password to be transferred to user securely
            else:
                st.error('Username not found')
        except Exception as e:
            st.error(e)

if authentication_status:

    # colored_header(label="Reset password",description=" ",color_name="violet-70")
    # with st.expander('Reset password'):
    #     try:
    #         if authenticator.reset_password(username, 'Reset password'):
    #             st.success('Password modified successfully')
    #     except Exception as e:
    #         st.error(e)


    colored_header(label="Try it now",description=" ",color_name="violet-70")

    col1, col2 = st.columns([2,2])
    with col1:
        P1 = card(
        title="DATA VISUALIZATION!",
        text="STEP 1",
        image= "https://th.bing.com/th/id/OIP.W-l_JOepX3wkql8lgeMsqgHaFQ?pid=ImgDet&rs=1")
        if P1:
            switch_page("DATA VISUALIZATION")

    with col2:
        P2 = card(
        title="FEATURE ENGINEERING!",
        text="STEP 2",
        image="https://th.bing.com/th/id/R.f859afb522ab770a6b1d670a75f54aec?rik=ENjw3pPepmHxFg&riu=http%3a%2f%2fimg.sccnn.com%2fbimg%2f338%2f13641.jpg&ehk=fcPxJs5SqNiKxLa1RcofksKgk86OHkghGx6%2fxoITceY%3d&risl=&pid=ImgRaw&r=0")
        if P2:
            switch_page("FEATURE ENGINEERING")
    

    P3 = card(
    title="PREDICTION!",
    text="STEP 3",
    image="https://th.bing.com/th/id/OIP.L0wnj6PnPxdmIdkaID23GwHaFY?pid=ImgDet&w=660&h=480&rs=1")
    if P3:
        switch_page("PREDICTION")

    colored_header(label="Jigsaw puzzle", description=" ",color_name="violet-70")

    
    P6 = card(
    title="SHAPley VALUE!",
    text="",
    image="https://th.bing.com/th/id/OIP.xwF5xKvzIRNrwmsvvNzntwHaEK?pid=ImgDet&w=1068&h=601&rs=1")
    if P6:
        switch_page("SHAPley value")

    col1, col2 = st.columns([2,2])  
    with col1:
        P4 = card(
        title="ACTIVE LEARNING!",
        text="",
        image="https://th.bing.com/th/id/OIP.B8_PwZbAvdAsGKZBFR6tMgHaEM?pid=ImgDet&w=530&h=300&rs=1")
        if P4:
            switch_page("ACTIVATE LEARNING")
    with col2:
        P7 = card(
        title="SYMBOLIC REGRESSION!",
        text="",
        image="")
        if P7:
            switch_page("SYMBOLIC REGRESSION")





    with open('./config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')


# streamlit_analytics.stop_tracking(unsafe_password="test123")

