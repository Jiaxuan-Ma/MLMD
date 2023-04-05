from utils import *


    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================

st.write('## DROP MISSING FEATUREs')
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
            null_columns = df.columns[df.isnull().any()]
            if len(null_columns) == 0:
                st.error('NO MISSING FEATUREs!')
                st.stop()
                
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

        colored_header(label="DROP MISSING FEATUREs",description=" ",color_name="violet-70")
    
        fs = FeatureSelector(features, targets)

        missing_threshold = st.slider("Missing Threshold",0.001, 1.0, 0.5)
        fs.identify_missing(missing_threshold)
        fs.features_dropped_missing = fs.features.drop(columns=fs.ops['missing'])
        
        data = pd.concat([fs.features_dropped_missing, targets], axis=1)
        st.write(data)
        tmp_download_link = download_button(data, f'features dropped missing.csv', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
        st.write('%d features with $\gt$ %0.2f missing values.\n' % (len(fs.ops['missing']), fs.missing_threshold))
        plot = customPlot()

        with st.expander('PLOT'):
            col1, col2 = st.columns([1,3])
            with col1:
                options_selected = [plot.set_title_fontsize(1),plot.set_label_fontsize(2),
                            plot.set_tick_fontsize(3),plot.set_legend_fontsize(4),plot.set_color('bin color',19,5)]
            with col2:
                plot.feature_missing(options_selected, fs.record_missing, fs.missing_stats)
        st.write('---')
        import streamlit as st
        import graphviz



elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


