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

st.write('## CLUSTER')
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
        colored_header(label="Data Information", description=" ", color_name="violet-70")
        with st.expander('Data Information'):
            df = pd.read_csv(file)

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
        
       #=============== cluster ================

        colored_header(label="Cluster",description=" ",color_name="violet-70")
        cluster = CLUSTER(features, targets)

        # colored_header(label="Choose Target", description=" ", color_name="violet-30")
        # target_selected_option = st.selectbox('target', list(cluster.targets)[::-1])

        # cluster.targets = targets[target_selected_option]

        model_path = './models/cluster'

        colored_header(label="Training", description=" ",color_name="violet-30")

        template_alg = model_platform(model_path)

        inputs, col2 = template_alg.show()
    
        if inputs['model'] == 'K-means':

            with col2:
                pass

            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                
                    cluster.model = KMeans(n_clusters=inputs['n clusters'], random_state = inputs['random state'])
                    
                    cluster.K_means()

                    clustered_df = pd.concat([cluster.features,pd.DataFrame(cluster.model.labels_)], axis=1)
                    
                    r_name='cluster label'
                    c_name=clustered_df.columns[-1]
                    
                    clustered_df.rename(columns={c_name:r_name},inplace=True)
                    st.write(clustered_df)
            
        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')