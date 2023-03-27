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
    
        fs = FeatureSelector(features,targets)

        target_selected_option = st.selectbox('target', list(fs.targets)[::-1])

        fs.targets = targets[target_selected_option]
        # st.write(fs.targets.head())
        colored_header(label="SELECTOR", description=" ",color_name="violet-30")

        model_path = './models/feature importance'
        
        template_alg = model_platform(model_path=model_path)

        colored_header(label="TRAINING", description=" ",color_name="violet-30")

        inputs, col2 = template_alg.show()
        # st.write(inputs)
        
        if inputs['model'] == 'LGBRegressor':
            
            fs.model = lgb.LGBMRegressor(n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'], verbose=-1)

            with col2:
                option_cumulative_importance = st.slider('cumulative importance',0.0, 1.0, 0.95)
            with st.container():
                button_train = st.button('train', use_container_width=True)
            if button_train:
                
                fs.lightGBM(eval_metric=inputs['metric'], n_iterations=inputs['niterations'], early_stopping= inputs['early stopping'], 
                                            test_size=inputs['test size'], round_number = inputs['early stopping rounds'])
                fs.identify_zero_low_importance(option_cumulative_importance)
                fs.feature_importance_select_show()

        # elif inputs['model'] == 'Permutation':

        #     st.write('you found me')

        # elif inputs['model'] == 'LinearRegressor':
            

        #     fs.model = SelectFromModel(estimator=LinearR())

        #     with col2:
        #         option_cumulative_importance = st.slider('cumulative importance',0.0, 1.0, 0.95)
        #     with st.container():
        #         button_train = st.button('train', use_container_width=True)
        #     if button_train:

        #         # fs.model.fit(fs.features, fs.targets)

        #         fs.LinearRegressor()
        #         # st.write(fs.feature_importances)
        #         fs.identify_zero_low_importance(option_cumulative_importance)
        #         fs.feature_importance_select_show()

        # elif inputs['model'] == 'LassoCV':
            
            
        #     fs.model = LassoCV()

        #     with col2:
        #         option_cumulative_importance = st.slider('cumulative importance',0.0, 1.0, 0.95)
            
        #     with st.container():
        #         button_train = st.button('train', use_container_width=True)
        #     if button_train:

        #         # fs.model.fit(fs.features, fs.targets)

        #         fs.LassoCV()

        #         fs.identify_zero_low_importance(option_cumulative_importance)
        #         fs.feature_importance_select_show()
        
        # elif inputs['model'] == 'ExtraTressClassifier':
            

        #     fs.model = ETR(n_estimators = inputs['nestimators'], 
        #                                 max_features=inputs['max features'],
        #                                 random_state=inputs['random state'])

        #     with col2:
        #         option_cumulative_importance = st.slider('cumulative importance',0.0, 1.0, 0.95)

        #     with st.container():
        #         button_train = st.button('train', use_container_width=True)
        #     if button_train:

        #         fs.model.fit(fs.features, fs.targets)

        #         fs.EXtraTreesClassifier()

        #         fs.identify_zero_low_importance(option_cumulative_importance)
        #         fs.feature_importance_select_show()


        # elif inputs['model'] == 'RFECVsvr':

        #     if inputs['kernel'] == 'linear':
        #         estimator = SVR(kernel = "linear")

        #     if inputs['kernel'] == 'rbf':
        #         estimator = SVR(kernel = 'rbf')

        #     fs.model = RFECV(estimator=estimator, step=inputs['step'],cv=inputs['cv'])
        #     with col2:
        #         option_cumulative_importance = st.slider('cumulative importance',0.0, 1.0, 0.95)

        #     with st.container():
        #         button_train = st.button('train', use_container_width=True)
        #     if button_train:

        #         # fs.model.fit(fs.features, fs.targets)

        #         fs.RFECVsvr()

        #         fs.identify_zero_low_importance(option_cumulative_importance)
        #         fs.feature_importance_select_show()

        elif inputs['model'] == 'RandomForestClassifier':
            
            fs.model = RFC()

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
                    plt.xlabel("feature importance")
                    plt.ylabel("r2")
                    st.pyplot(fig)

        elif inputs['model'] == 'RandomForestRegressor':
                    
                    fs.model = RFR()

                    with col2:
                        
                        option_cumulative_importance = st.slider('cumulative importance',0.5, 1.0, 0.95)
                        Embedded_method = st.checkbox('Embedded method',False)
                        if Embedded_method:
                            cv = st.number_input('cv',1,10,5)
                          
                            feature_importance_max =np.float((fs.model.fit(fs.features, fs.targets).feature_importances_).max())
                            range_threshold = st.slider('threshold range',0.0, feature_importance_max,(0.0, feature_importance_max))
            
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
                            plt.xlabel("feature importance")
                            plt.ylabel("r2")
                            st.pyplot(fig)

        st.write('---')


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


    # 统计网页刷新数据和脚本部件运行数据
    # streamlit_analytics.stop_tracking(unsafe_password="test123")

