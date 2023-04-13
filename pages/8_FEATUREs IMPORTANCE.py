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

st.write('## FEATURE IMPORTANCE')
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
    
        fs = FeatureSelector(features,targets)

        colored_header(label="Choose Target", description=" ", color_name="violet-30")

        target_selected_option = st.selectbox('target', list(fs.targets)[::-1])

        fs.targets = targets[target_selected_option]
        # st.write(fs.targets.head())
        colored_header(label="Selector", description=" ",color_name="violet-30")

        model_path = './models/feature importance'
        
        template_alg = model_platform(model_path=model_path)

        colored_header(label="Training", description=" ",color_name="violet-30")

        inputs, col2 = template_alg.show()
        # st.write(inputs)

        if inputs['model'] == 'LinearRegressor':
            

            fs.model = LinearR()

            with col2:
                option_cumulative_importance = st.slider('cumulative importance',0.0, 1.0, 0.95)
                Embedded_method = st.checkbox('Embedded method',False)
                if Embedded_method:
                    cv = st.number_input('cv',1,10,5)
            with st.container():
                button_train = st.button('train', use_container_width=True)
            if button_train:

                fs.LinearRegressor()     
                fs.identify_zero_low_importance(option_cumulative_importance)
                fs.feature_importance_select_show()
                if Embedded_method:
                    
                    threshold  = fs.cumulative_importance

                    feature_importances = fs.feature_importances.set_index('feature',drop = False)

                    features = []
                    scores = []
                    cumuImportance = []
                    for i in range(1, len(fs.features.columns) + 1):
                        features.append(feature_importances.iloc[:i, 0].values.tolist())
                        X_selected = fs.features[features[-1]]
                        score = CVS(fs.model, X_selected, fs.targets, cv=cv).mean()

                        cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                        scores.append(score)
                    cumu_importance = np.array(cumuImportance)
                    scores = np.array(scores) 
                    fig, ax = plt.subplots()
                    ax = plt.plot(cumu_importance, scores,'o-')
                    plt.xlabel("cumulative feature importance")
                    plt.ylabel("r2")
                    st.pyplot(fig)
        elif inputs['model'] == 'LassoRegressor':
            
            fs.model = Lasso(random_state=inputs['random state'])

            with col2:
                option_cumulative_importance = st.slider('cumulative importance',0.0, 1.0, 0.95)
                Embedded_method = st.checkbox('Embedded method',False)
                if Embedded_method:
                    cv = st.number_input('cv',1,10,5)

            with st.container():
                button_train = st.button('train', use_container_width=True)
            if button_train:

                fs.LassoRegressor()

                fs.identify_zero_low_importance(option_cumulative_importance)
                fs.feature_importance_select_show()
                if Embedded_method:
                    
                    threshold  = fs.cumulative_importance

                    feature_importances = fs.feature_importances.set_index('feature',drop = False)

                    features = []
                    scores = []
                    cumuImportance = []
                    for i in range(1, len(fs.features.columns) + 1):
                        features.append(feature_importances.iloc[:i, 0].values.tolist())
                        X_selected = fs.features[features[-1]]
                        score = CVS(fs.model, X_selected, fs.targets, cv=cv).mean()

                        cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                        scores.append(score)
                    cumu_importance = np.array(cumuImportance)
                    scores = np.array(scores) 
                    fig, ax = plt.subplots()
                    ax = plt.plot(cumu_importance, scores,'o-')
                    plt.xlabel("cumulative feature importance")
                    plt.ylabel("r2")
                    st.pyplot(fig)

        elif inputs['model'] == 'RidgeRegressor':
            

            fs.model = Ridge(random_state=inputs['random state'])

            with col2:
                option_cumulative_importance = st.slider('cumulative importance',0.0, 1.0, 0.95)
                Embedded_method = st.checkbox('Embedded method',False)
                if Embedded_method:
                    cv = st.number_input('cv',1,10,5)
            with st.container():
                button_train = st.button('train', use_container_width=True)
            if button_train:
                fs.RidgeRegressor()     
                fs.identify_zero_low_importance(option_cumulative_importance)
                fs.feature_importance_select_show()
                if Embedded_method:
                    
                    threshold  = fs.cumulative_importance

                    feature_importances = fs.feature_importances.set_index('feature',drop = False)

                    features = []
                    scores = []
                    cumuImportance = []
                    for i in range(1, len(fs.features.columns) + 1):
                        features.append(feature_importances.iloc[:i, 0].values.tolist())
                        X_selected = fs.features[features[-1]]
                        score = CVS(fs.model, X_selected, fs.targets, cv=cv).mean()

                        cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                        scores.append(score)
                    cumu_importance = np.array(cumuImportance)
                    scores = np.array(scores) 
                    fig, ax = plt.subplots()
                    ax = plt.plot(cumu_importance, scores,'o-')
                    plt.xlabel("cumulative feature importance")
                    plt.ylabel("r2")
                    st.pyplot(fig)
        elif inputs['model'] == 'LassoRegressor':
            
            fs.model = Lasso(random_state=inputs['random state'])

            with col2:
                option_cumulative_importance = st.slider('cumulative importance',0.0, 1.0, 0.95)
                Embedded_method = st.checkbox('Embedded method',False)
                if Embedded_method:
                    cv = st.number_input('cv',1,10,5)
            with st.container():
                button_train = st.button('train', use_container_width=True)
            if button_train:

                fs.LassoRegressor()

                fs.identify_zero_low_importance(option_cumulative_importance)
                fs.feature_importance_select_show()
                if Embedded_method:
                    
                    threshold  = fs.cumulative_importance

                    feature_importances = fs.feature_importances.set_index('feature',drop = False)

                    features = []
                    scores = []
                    cumuImportance = []
                    for i in range(1, len(fs.features.columns) + 1):
                        features.append(feature_importances.iloc[:i, 0].values.tolist())
                        X_selected = fs.features[features[-1]]
                        score = CVS(fs.model, X_selected, fs.targets, cv=cv).mean()

                        cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                        scores.append(score)
                    cumu_importance = np.array(cumuImportance)
                    scores = np.array(scores) 
                    fig, ax = plt.subplots()
                    ax = plt.plot(cumu_importance, scores,'o-')
                    plt.xlabel("cumulative feature importance")
                    plt.ylabel("r2")
                    st.pyplot(fig)

        elif inputs['model'] == 'RandomForestClassifier':
            
            fs.model =  RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],
                            min_samples_leaf=inputs['min samples leaf'], min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'], warm_start=inputs['warm start'])
                    
            st.write(fs.model)
            with col2:
                
                option_cumulative_importance = st.slider('cumulative importance',0.5, 1.0, 0.95)
                Embedded_method = st.checkbox('Embedded method',False)
                if Embedded_method:
                    cv = st.number_input('cv',1,10,5)


            with st.container():
                button_train = st.button('train', use_container_width=True)
            if button_train:

                fs.RandomForestClassifier()

                fs.identify_zero_low_importance(option_cumulative_importance)
                fs.feature_importance_select_show()

                if Embedded_method:
                    
                    threshold  = fs.cumulative_importance

                    feature_importances = fs.feature_importances.set_index('feature',drop = False)

                    features = []
                    scores = []
                    cumuImportance = []
                    for i in range(1, len(fs.features.columns) + 1):
                        features.append(feature_importances.iloc[:i, 0].values.tolist())
                        X_selected = fs.features[features[-1]]
                        score = CVS(fs.model, X_selected, fs.targets, cv=cv).mean()

                        cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                        scores.append(score)
                    cumu_importance = np.array(cumuImportance)
                    scores = np.array(scores) 
                    fig, ax = plt.subplots()
                    ax = plt.plot(cumu_importance, scores,'o-')
                    plt.xlabel("cumulative feature importance")
                    plt.ylabel("r2")
                    st.pyplot(fig)

        elif inputs['model'] == 'RandomForestRegressor':
                    
                    fs.model = RFR(criterion = inputs['criterion'], n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                    min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                    n_jobs=inputs['njobs'])
                    st.write(fs.model)
                    with col2:
                        
                        option_cumulative_importance = st.slider('cumulative importance',0.5, 1.0, 0.95)
                        Embedded_method = st.checkbox('Embedded method',False)
                        if Embedded_method:
                            cv = st.number_input('cv',1,10,5)
                          
                            # feature_importance_max =np.float((fs.model.fit(fs.features, fs.targets).feature_importances_).max())
                            # range_threshold = st.slider('threshold range',0.0, feature_importance_max,(0.0, feature_importance_max))
            
                    with st.container():
                        button_train = st.button('train', use_container_width=True)
                    if button_train:

                        fs.RandomForestRegressor()

                        fs.identify_zero_low_importance(option_cumulative_importance)
                        fs.feature_importance_select_show()

                        if Embedded_method:
                            
                            threshold  = fs.cumulative_importance

                            feature_importances = fs.feature_importances.set_index('feature',drop = False)

                            features = []
                            scores = []
                            cumuImportance = []
                            for i in range(1, len(fs.features.columns) + 1):
                                features.append(feature_importances.iloc[:i, 0].values.tolist())
                                X_selected = fs.features[features[-1]]
                                score = CVS(fs.model, X_selected, fs.targets, cv=cv).mean()

                                cumuImportance.append(feature_importances.loc[features[-1][-1], 'cumulative_importance'])
                                scores.append(score)
                            cumu_importance = np.array(cumuImportance)
                            scores = np.array(scores) 
                            fig, ax = plt.subplots()
                            ax = plt.plot(cumu_importance, scores,'o-')
                            plt.xlabel("cumulative feature importance")
                            plt.ylabel("r2")
                            st.pyplot(fig)

        st.write('---')


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


    # 统计网页刷新数据和脚本部件运行数据
    # streamlit_analytics.stop_tracking(unsafe_password="test123")

