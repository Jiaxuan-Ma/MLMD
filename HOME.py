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
# MACHINE LEARNING FOR MATERIALs
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
        title="DATA PRELIMINARY!",
        text="STEP 1",
        image= "https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P1:
            switch_page("DATA PRELIMINARY")

    with col2:
        P2 = card(
        title="FEATURE ENGINEERING!",
        text="STEP 2",
        image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P2:
            switch_page("FEATURE ENGINEERING")
    

    P3 = card(
    title="PREDICTION!",
    text="STEP 3",
    image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
    if P3:
        switch_page("PREDICTION")

    colored_header(label="Jigsaw puzzle", description=" ",color_name="violet-70")

    
    P6 = card(
    title="SHAPley VALUE!",
    text="",
    image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
    if P6:
        switch_page("SHAPley value")

    col1, col2 = st.columns([2,2])  
    with col1:
        P4 = card(
        title="ACTIVE LEARNING!",
        text="",
        image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P4:
            switch_page("ACTIVATE LEARNING")
    with col2:
        P7 = card(
        title="SYMBOLIC REGRESSION!",
        text="",
        image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
        if P7:
            switch_page("SYMBOLIC REGRESSION")

    P8 = card(
    title="ENSEMBLE LEARNING!",
    text="",
    image="https://img1.imgtp.com/2023/04/08/L3wPy8Ra.png")
    if P8:
        switch_page("ENSEMBLE LEARNING")

    with open('./config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')


# streamlit_analytics.stop_tracking(unsafe_password="test123")

