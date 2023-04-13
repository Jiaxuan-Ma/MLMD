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

st.write('## DROP NUNIQUE FEATUREs')
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
        
       #=============== drop nunqiue features ================

        colored_header(label="Drop Nunique Features",description=" ",color_name="violet-70")
        fs = FeatureSelector(features, targets)
        plot = customPlot() 

        col1, col2 = st.columns([1,3])
        with col1:
            
            fs.identify_nunique()
            option_counts = st.slider('drop unique counts',0, int(fs.unique_stats.max())-1,1)
            st.write(fs.unique_stats)
        with col2:

            fs.identify_nunique(option_counts)
            fs.features_dropped_single = fs.features.drop(columns=fs.ops['single_unique'])
            data = pd.concat([fs.features_dropped_single, targets], axis=1)
            st.write(fs.features_dropped_single)
            
            tmp_download_link = download_button(data, f'features dropped nunique.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
            st.write('%d features $\leq$  %d unique value.\n' % (len(fs.ops['single_unique']),option_counts))
   
        with st.expander('Plot'):
            col1, col2 = st.columns([1,3])
            with col1:
                options_selected = [plot.set_title_fontsize(6),plot.set_label_fontsize(7),
                            plot.set_tick_fontsize(8),plot.set_legend_fontsize(9),plot.set_color('bin color',19,10)]
            with col2:
                plot.feature_nunique(options_selected, fs.record_single_unique,fs.unique_stats)     
            
        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


