from utils import *


st.set_page_config(
    page_title="MLMD",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
    }
)

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ======================================================

st.write('## DROP COLLINEAR FEATUREs')
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

        colored_header(label="Data Information",description=" ",color_name="violet-70")
        with st.expander('Data Information'):
            df = pd.read_csv(file)
            check_string_NaN(df)
            colored_header(label="Data", description=" ",color_name="blue-70")
            nrow = st.slider("rows", 1, len(df)-1, 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="Features vs Targets",description=" ",color_name="blue-30")

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
        
        colored_header(label="Drop Collinear Features",description=" ",color_name="violet-30")
        fs = FeatureSelector(features, targets)
        plot = customPlot() 

        target_selected_option = st.selectbox('choose target', list(fs.targets))
        target_selected = fs.targets[target_selected_option]

        col1, col2 = st.columns([1,3])
        with col1:
            corr_method = st.selectbox("correlation method",["pearson","spearman","kendall"])
            correlation_threshold = st.slider("correlation threshold",0.001, 1.0, 0.9) # 0.52
            corr_matrix = pd.concat([fs.features, target_selected], axis=1).corr(corr_method)
            fs.identify_collinear(corr_matrix, correlation_threshold)
            fs.judge_drop_f_t_after_f_f([target_selected_option], corr_matrix)

            is_mask = st.selectbox('is mask',('Yes', 'No'))
            with st.expander('Plot Parameters'):
                options_selected = [plot.set_title_fontsize(19),plot.set_label_fontsize(20),
                                    plot.set_tick_fontsize(21),plot.set_legend_fontsize(22)]
            with st.expander('Dropped Features'):
                st.write(fs.record_collinear)
        with col2:
            fs.features_dropped_collinear = fs.features.drop(columns=fs.ops['collinear'])
            assert fs.features_dropped_collinear.size != 0,'zero feature !' 
            corr_matrix_drop_collinear = fs.features_dropped_collinear.corr(corr_method)
            plot.corr_cofficient(options_selected, is_mask, corr_matrix_drop_collinear)
            with st.expander('Processed Data'):
                data = pd.concat([fs.features_dropped_collinear, targets], axis=1)
                st.write(data)
                tmp_download_link = download_button(data, f'features dropped collinear feature.csv', button_text='download')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
        
        st.write('---')
        
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


