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

st.write('## DATA ENHENCEMENT')
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
        with st.expander('DATA INFORMATION'):
            df = pd.read_csv(file)
            # ============ check NaN ===========
            check_string_NaN(df)
                
            colored_header(label="DATA", description=" ",color_name="blue-70")
            nrow = st.slider("rows", 1, len(df)-1, 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="FEATUREs SELECT",description=" ",color_name="blue-30")

            target_num = st.number_input('input target',  min_value=1, max_value=10, value=1)
            st.write('target number', target_num)
            
            col_feature, col_target = st.columns(2)
            
            # features
            features = df.iloc[:,:-target_num]
            # targets
            targets = df.iloc[:,-target_num:]
            with col_feature:    
                st.write(features.head())
            with col_target:   
                st.write(targets.head())
        
        #=============== drop major missing features ================

        colored_header(label="DATA ENHENCEMENT",description=" ",color_name="violet-70")

        st.write('demo for unbalanced binary class')
        fs = FeatureSelector(features, targets)

        sm = SMOTE(random_state=42)
        # fs.features, fs.targets = sm.fit_resample(fs.features, fs.targets)
        # n_sample = fs.features.shape[0]
        # pd.Series(fs.targets).value_counts(normalize=True)
        # n_1_sample = pd.Series(fs.targets).value_counts()[1]
        # n_0_sample = pd.Series(fs.targets).value_counts()[0]
        # st.write('样本个数：{}; 1 占{:.2%};0占{:.2%}'.format(n_sample,n_1_sample/n_sample,n_0_sample/n_sample))
   
        st.write('---')
        import streamlit as st
        import graphviz



elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')