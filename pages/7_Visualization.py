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
        check_string_NaN(df)
        
        colored_header(label="Data Information",description=" ",color_name="violet-70")

        nrow = st.slider("rows", 1, len(df)-1, 5)
        df_nrow = df.head(nrow)
        st.write(df_nrow)

        colored_header(label="Data Statistics",description=" ",color_name="violet-30")

        st.write(df.describe())

        tmp_download_link = download_button(df.describe(), f'data statistics.csv', button_text='download')
        
        st.markdown(tmp_download_link, unsafe_allow_html=True)

        colored_header(label="Features vs Targets", description=" ",color_name="violet-70")

        target_num = st.number_input('input target',  min_value=1, max_value=10, value=1)
        st.write('target number', target_num)
        col_feature, col_target = st.columns(2)
        st.write('---')
        # features
        features = df.iloc[:,:-target_num]
        # targets
        targets = df.iloc[:,-target_num:]
        with col_feature:    
            st.write(features.head())
        with col_target:   
            st.write(targets.head())
        # =========== Features visulization =================

        # st.write('### Feature statistics')
        colored_header(label="Feature Statistics Distribution", description=" ",color_name="violet-30")

        feature_selected_name = st.selectbox('feature list',list(features))
    
        feature_selected_value = features[feature_selected_name]
        plot = customPlot()
        col1, col2 = st.columns([1,3])
        with col1:  
            with st.expander("Plot parameters"):
                options_selected = [plot.set_title_fontsize(1),plot.set_label_fontsize(2),
                            plot.set_tick_fontsize(3),plot.set_legend_fontsize(4),plot.set_color('line color',6,5),plot.set_color('bin color',0,6)]
        with col2:
            plot.feature_hist_kde(options_selected,feature_selected_name,feature_selected_value)

        #=========== Targets visulization ==================

        colored_header(label="Target Statistics Distribution", description=" ",color_name="violet-30")

        target_selected_name = st.selectbox('target list',list(targets))

        target_selected_value = targets[target_selected_name]
        plot = customPlot()
        col1, col2 = st.columns([1,3])
        with col1:  
            with st.expander("Plot parameters"):
                options_selected = [plot.set_title_fontsize(7),plot.set_label_fontsize(8),
                            plot.set_tick_fontsize(9),plot.set_legend_fontsize(10), plot.set_color('line color',6,11), plot.set_color('bin color',0,12)]
        with col2:
            plot.target_hist_kde(options_selected,target_selected_name,target_selected_value)

        #=========== Features analysis ==================

        colored_header(label="Feature Recipe Distribution", description=" ",color_name="violet-30")

        feature_range_selected_name = st.slider('Selected different type of feature',1,len(features.columns), (1,2))
        min_feature_selected = feature_range_selected_name[0]-1
        max_feature_selected = feature_range_selected_name[1]
        feature_range_selected_value = features.iloc[:,min_feature_selected: max_feature_selected]
        data_by_feature_type = df.groupby(list(feature_range_selected_value))
        feature_type_data = create_data_with_group_and_counts(data_by_feature_type)
        IDs = [str(id_) for id_ in feature_type_data['ID']]
        Counts = feature_type_data['Count']
        col1, col2 = st.columns([1,3])
        with col1:  
            with st.expander("Plot parameters"):
                options_selected = [plot.set_title_fontsize(13),plot.set_label_fontsize(14),
                            plot.set_tick_fontsize(15),plot.set_legend_fontsize(16),plot.set_color('bin color',0, 17)]
        with col2:
            plot.featureSets_statistics_hist(options_selected,IDs, Counts)

        colored_header(label="Distribution of Feature in Dataset", description=" ",color_name="violet-30")
        feature_selected_name = st.selectbox('feature', list(features))
        feature_selected_value = features[feature_selected_name]
        col1, col2 = st.columns([1,3])
        with col1:  
            with st.expander("Plot parameters"):
                options_selected = [plot.set_title_fontsize(18),plot.set_label_fontsize(19),
                            plot.set_tick_fontsize(20),plot.set_legend_fontsize(21), plot.set_color('bin color', 0, 22)]
        with col2:
            plot.feature_distribution(options_selected,feature_selected_name,feature_selected_value)

        colored_header(label="Features and Targets", description=" ",color_name="violet-30")
        col1, col2 = st.columns([1,3])
        with col1:  
            with st.expander("Plot parameters"):
                options_selected = [plot.set_title_fontsize(23),plot.set_label_fontsize(24),
                            plot.set_tick_fontsize(25),plot.set_legend_fontsize(26),plot.set_color('scatter color',0, 27),plot.set_color('line color',6,28)]
        with col2:
            plot.features_and_targets(options_selected,df, list(features), list(targets))
        
        # st.write("### Targets and Targets ")
        if targets.shape[1] != 1:
            colored_header(label="Targets and Targets", description=" ",color_name="violet-30")
            col1, col2 = st.columns([1,3])
            with col1:  
                with st.expander("Plot parameters"):
                    options_selected = [plot.set_title_fontsize(29),plot.set_label_fontsize(30),
                                plot.set_tick_fontsize(31),plot.set_legend_fontsize(32),plot.set_color('scatter color',0, 33),plot.set_color('line color',6,34)]
            with col2:
                plot.targets_and_targets(options_selected,df, list(targets))
    st.write('---')


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# streamlit_analytics.stop_tracking(unsafe_password="test123")