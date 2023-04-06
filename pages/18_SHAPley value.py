from utils import *


    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================

st.write('## Shap value analysis')
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
        

        colored_header(label="Shapley value",description=" ",color_name="violet-70")

        fs = FeatureSelector(features, targets)
        # regressor = st.selectbox('tree',['linear','kernel','sampling'])
        reg = XGBR()
        X_train, X_test, y_train, y_test = TTS(fs.features, fs.targets, random_state=0) 
        test_size = st.slider('test size',0.1, 0.5, 0.2) 
        random_state = st.checkbox('random state 1024',True)
        if random_state:
            random_state = 1024
        else:
            random_state = None
            
        fs.Xtrain,fs.Xtest, fs.Ytrain, fs.Ytest = TTS(fs.features,fs.targets,test_size=test_size,random_state=random_state)
        reg.fit(fs.Xtrain, fs.Ytrain)

        explainer = shap.TreeExplainer(reg)
        
        shap_values = explainer(fs.features)

        colored_header(label="SHAP Feature Importance", description=" ",color_name="violet-30")
        nfeatures = st.slider("features", 2, fs.features.shape[1],fs.features.shape[1])
        st_shap(shap.plots.bar(shap_values, max_display=nfeatures))

        colored_header(label="SHAP Feature Cluster", description=" ",color_name="violet-30")
        clustering = shap.utils.hclust(fs.features, fs.targets)
        clustering_cutoff = st.slider('clustering cutoff', 0.0,1.0,0.5)
        nfeatures = st.slider("features", 2, fs.features.shape[1],fs.features.shape[1], key=2)
        st_shap(shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=clustering_cutoff, max_display=nfeatures))

        colored_header(label="SHAP Beeswarm", description=" ",color_name="violet-30")
        rank_option = st.selectbox('rank option',['max','mean'])
        max_dispaly = st.slider('max display',2, fs.features.shape[1],fs.features.shape[1])
        if rank_option == 'max':
            st_shap(shap.plots.beeswarm(shap_values, order = shap_values.abs.max(0), max_display =max_dispaly))
        else:
            st_shap(shap.plots.beeswarm(shap_values, order = shap_values.abs.mean(0), max_display =max_dispaly))

        colored_header(label="SHAP Dependence", description=" ",color_name="violet-30")
        
        # shap.dependence_plot('Si', shap_values, X[cols], interaction_index='Mn', show=False)
        shap_values = explainer.shap_values(fs.features) 
        list_features = fs.features.columns.tolist()
        feature = st.selectbox('feature',list_features)
        interact_feature = st.selectbox('interact feature', list_features)
        st_shap(shap.dependence_plot(feature, shap_values, fs.features, display_features=fs.features,interaction_index=interact_feature))
        
        colored_header(label="SHAP Most Important Feature", description=" ",color_name="violet-30")
        shap_values = explainer(fs.features)
        ind_mean = shap_values.abs.mean(0).argsort[-1]

        ind_max = shap_values.abs.max(0).argsort[-1]

        ind_perc = shap_values.abs.percentile(95, 0).argsort[-1]
        st_shap(shap.plots.scatter(shap_values[:, ind_mean]))


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


