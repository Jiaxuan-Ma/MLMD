from utils import *

# streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================


st.write('## Data Profiling')
st.write('---')

# =====================================================
if st.session_state["authentication_status"]:

    file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed")
    upload_file = option_menu(None, ["Upload"], icons=['cloud-upload'], menu_icon="cast", 
                              default_index=0, orientation="horizontal",styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "25px"}, 
                "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "gray"}})  
            
    if file is not None:
        df = pd.read_csv(file)
        # =============== check string / NaN ====================== 
        
        colored_header(label="DATA TABLE",description=" ",color_name="violet-70")

        nrow = st.slider("rows", 1, len(df)-1, 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="REPORT",description=" ",color_name="violet-30")

        pr = df.profile_report()

        st_profile_report(pr)

    st.write('---')


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# streamlit_analytics.stop_tracking(unsafe_password="test123")