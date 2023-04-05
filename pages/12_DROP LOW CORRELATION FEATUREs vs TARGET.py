from utils import *


    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================

st.write('## DROP LOW CORRELATION FEATUREs vs TARGET')
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
            check_string_NaN(df)
            colored_header(label="TABLE", description=" ",color_name="blue-70")
            nrow = st.slider("rows", 1, len(df)-1, 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="FEATUREs vs TARGETs",description=" ",color_name="blue-30")

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
        
       #=============== drop nunqiue features ================

        colored_header(label="DROP LOW CORRELATION FEATUREs vs TARGET",description=" ",color_name="violet-70")
        fs = FeatureSelector(features, targets)
        plot = customPlot() 

        col1, col2 = st.columns([1,3])
        # according the feature and target correlation to drop feature
        with col1:  
            corr_method = st.selectbox("correlation method",["pearson","spearman","kendall","MIR"], key=15)  
            if corr_method != "MIR":
                option_dropped_threshold = st.slider('corr threshold f_t',0.0, 1.0,0.0)
            if corr_method == 'MIR':
                options_seed = st.checkbox('random state 1024',True)
            with st.expander('PLOT PARAMETERS'):
                options_selected = [plot.set_title_fontsize(11),plot.set_label_fontsize(12),
                    plot.set_tick_fontsize(13),plot.set_legend_fontsize(14),plot.set_color('bin color',19,16)]
            
        with col2:
            
            target_selected_option = st.selectbox('choose target', list(fs.targets))

            target_selected = fs.targets[target_selected_option]
            if corr_method != "MIR":
                corr_matrix = pd.concat([fs.features, target_selected], axis=1).corr(corr_method).abs()

                fs.judge_drop_f_t([target_selected_option], corr_matrix, option_dropped_threshold)
                
                fs.features_dropped_f_t = fs.features.drop(columns=fs.ops['f_t_low_corr'])
                corr_f_t = pd.concat([fs.features_dropped_f_t, target_selected], axis=1).corr(corr_method)[target_selected_option][:-1]

                plot.corr_feature_target(options_selected, corr_f_t)
                with st.expander('PROCESSED DATA'):
                    data = pd.concat([fs.features_dropped_f_t, targets], axis=1)
                    st.write(data)
                    tmp_download_link = download_button(data, f'features dropped low corr feature.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
            else:
                if options_seed:
                    corr_mir  = MIR(fs.features, target_selected, random_state=1024)
                else:
                    corr_mir = MIR(fs.features, target_selected)
                corr_mir = pd.DataFrame(corr_mir).set_index(pd.Index(list(fs.features.columns)))
                corr_mir.rename(columns={0: 'mutual info'}, inplace=True)
                plot.corr_feature_target_mir(options_selected, corr_mir)
 
        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


