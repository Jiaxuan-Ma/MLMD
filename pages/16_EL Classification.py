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

st.write('## ENSEMBLE CLASSIFICATION')
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
    
        clf = CLASSIFIER(features,targets)

        colored_header(label="Choose Target", description=" ", color_name="violet-30")
        target_selected_option = st.selectbox('target', list(clf.targets)[::-1])

        clf.targets = targets[target_selected_option]

        # st.write(fs.targets.head())
        colored_header(label="Classifier", description=" ",color_name="violet-30")

        model_path = './models/el classifiers'
        
        template_alg = model_platform(model_path)

        colored_header(label="Training", description=" ",color_name="violet-30")

        inputs, col2 = template_alg.show()
       
        if inputs['model'] == 'BaggingClassifier':

            with col2:
                with st.expander('Operator'):
                      
                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])
                    
                    data_process = st.selectbox('data process', ('train test split','cross val score'))
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        cv = st.number_input('cv',1,10,5)
         
            
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
                  
                    if inputs['base estimator'] == "DecisionTree":    
                        clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                      max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
   
                        clf.BaggingClassifier()
                        
                        plot = customPlot()
                        cm = confusion_matrix(y_true=clf.Ytest, y_pred=clf.Ypred)
                        plot.confusion_matrix(cm)
                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                                            
                    elif inputs['base estimator'] == "SupportVector": 
                        clf.model = BaggingClassifier(estimator =SVC(), n_estimators=inputs['nestimators'], max_features=inputs['max features'])    

                        clf.BaggingClassifier()
                        
                        plot = customPlot()
                        cm = confusion_matrix(y_true=clf.Ytest, y_pred=clf.Ypred)
                        plot.confusion_matrix(cm)
                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True) 
                    
                    elif inputs['base estimator'] == "LogisticRegression":    
                        
                        clf.model = BaggingClassifier(estimator = LR(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                      max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1)  
                        clf.BaggingClassifier()
                        
                        plot = customPlot()
                        cm = confusion_matrix(y_true=clf.Ytest, y_pred=clf.Ypred)
                        plot.confusion_matrix(cm)
                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True) 

                elif data_process == 'cross val score':

                    if inputs['base estimator'] == "DecisionTree":    
                        clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                      max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                        cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                        st.info('cv mean accuracy score: {}'.format(cvs.mean())) 

                    elif inputs['base estimator'] == "SupportVector": 
                        clf.model = BaggingClassifier(estimator =SVC(), n_estimators=inputs['nestimators'], max_features=inputs['max features'])    
                        cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                        st.info('cv mean accuracy score: {}'.format(cvs.mean())) 
                        
                    elif inputs['base estimator'] == "LogisticRegression": 
                        clf.model = BaggingClassifier(estimator = LR(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1)                         
                        cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                        st.info('cv mean accuracy score: {}'.format(cvs.mean())) 
        
        if inputs['model'] == 'AdaBoostClassifier':

            with col2:
                with st.expander('Operator'):
                      
                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])
                    
                    data_process = st.selectbox('data process', ('train test split','cross val score'))
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        cv = st.number_input('cv',1,10,5)
         
            
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
                  
                    if inputs['base estimator'] == "DecisionTree":    
                        clf.model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
   
                        clf.AdaBoostClassifier()
                        
                        plot = customPlot()
                        cm = confusion_matrix(y_true=clf.Ytest, y_pred=clf.Ypred)
                        plot.confusion_matrix(cm)
                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                                            
                    elif inputs['base estimator'] == "SupportVector": 
                        
                        clf.model = AdaBoostClassifier(estimator=SVC(),algorithm='SAMME', n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
   
                        clf.AdaBoostClassifier()
                        
                        plot = customPlot()
                        cm = confusion_matrix(y_true=clf.Ytest, y_pred=clf.Ypred)
                        plot.confusion_matrix(cm)
                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True) 
                    
                    elif inputs['base estimator'] == "LogisticRegression":    
                        
                        clf.model = AdaBoostClassifier(estimator=LR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
   
                        clf.AdaBoostClassifier()
                        
                        plot = customPlot()
                        cm = confusion_matrix(y_true=clf.Ytest, y_pred=clf.Ypred)
                        plot.confusion_matrix(cm)
                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True) 

                elif data_process == 'cross val score':

                    if inputs['base estimator'] == "DecisionTree":    
                        clf.model = BaggingClassifier(estimator = tree.DecisionTreeClassifier(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                      max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                        cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                        st.info('cv mean accuracy score: {}'.format(cvs.mean())) 

                    elif inputs['base estimator'] == "SupportVector": 
                        clf.model = BaggingClassifier(estimator =SVC(), n_estimators=inputs['nestimators'], max_features=inputs['max features'])    
                        cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                        st.info('cv mean accuracy score: {}'.format(cvs.mean())) 
                        
                    elif inputs['base estimator'] == "LogisticRegression": 
                        clf.model = BaggingClassifier(estimator = LR(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                    max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1)                         
                        cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                        st.info('cv mean accuracy score: {}'.format(cvs.mean())) 
        
        if inputs['model'] == 'GradientBoostingClassifier':

            with col2:
                with st.expander('Operator'):
                      
                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])
                    
                    data_process = st.selectbox('data process', ('train test split','cross val score'))
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        cv = st.number_input('cv',1,10,5)
         
            
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
                  
                        clf.model = GradientBoostingClassifier(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_depth=inputs['max depth'],max_features=inputs['max features'],
                                                               random_state=inputs['random state'])
                        clf.GradientBoostingClassifier()
                        
                        plot = customPlot()
                        cm = confusion_matrix(y_true=clf.Ytest, y_pred=clf.Ypred)
                        plot.confusion_matrix(cm)
                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif data_process == 'cross val score':
                        
                        clf.model = GradientBoostingClassifier(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_depth=inputs['max depth'],max_features=inputs['max features'],
                                                               random_state=inputs['random state'])
   
                        cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                        st.info('cv mean accuracy score: {}'.format(cvs.mean())) 

        if inputs['model'] == 'XGBClassifier':

            with col2:
                with st.expander('Operator'):
                      
                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])
                    
                    data_process = st.selectbox('data process', ('train test split','cross val score'))
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        cv = st.number_input('cv',1,10,5)
         
            
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
                  
                        clf.model = xgb.XGBClassifier(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                      max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                      learning_rate=inputs['learning rate'])
                        clf.XGBClassifier()
                        
                        plot = customPlot()
                        cm = confusion_matrix(y_true=clf.Ytest, y_pred=clf.Ypred)
                        plot.confusion_matrix(cm)
                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif data_process == 'cross val score':
                        
                        clf.model = xgb.XGBClassifier(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                      max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                      learning_rate=inputs['learning rate'])
   
                        cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                        st.info('cv mean accuracy score: {}'.format(cvs.mean())) 
        
        if inputs['model'] == 'LGBMClassifier':

            with col2:
                with st.expander('Operator'):
                      
                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])
                    
                    data_process = st.selectbox('data process', ('train test split','cross val score'))
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        cv = st.number_input('cv',1,10,5)
         
            
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
                  
                    clf.model = lgb.LGBMClassifier(niterations=inputs['niterations'],nestimators=inputs['nestimators'],learning_rate=inputs['learning rate'],
                                                    num_leaves=inputs['num_leaves'],max_depth=inputs['max depth'])

                    clf.LGBMClassifier()
                    
                    plot = customPlot()
                    cm = confusion_matrix(y_true=clf.Ytest, y_pred=clf.Ypred)
                    plot.confusion_matrix(cm)
                    result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif data_process == 'cross val score':
                        
                    clf.model = lgb.LGBMClassifier(niterations=inputs['niterations'],nestimators=inputs['nestimators'],learning_rate=inputs['learning rate'],
                                                    num_leaves=inputs['num_leaves'],max_depth=inputs['max depth'])

                    cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                    st.info('cv mean accuracy score: {}'.format(cvs.mean()))        

        if inputs['model'] == 'CatBoostClassifier':

            with col2:
                with st.expander('Operator'):
                      
                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])
                    
                    data_process = st.selectbox('data process', ('train test split','cross val score'))
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif data_process == 'cross val score':

                        if preprocess == 'StandardScaler':
                            clf.features = StandardScaler().fit_transform(clf.features)
                        if preprocess == 'MinMaxScaler':
                            clf.features = MinMaxScaler().fit_transform(clf.features)
                        cv = st.number_input('cv',1,10,5)
         
            
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            
            if button_train:
                if data_process == 'train test split':
                  
                        clf.model = CatBoostClassifier(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'])

                        clf.CatBoostClassifier()
                        
                        plot = customPlot()
                        cm = confusion_matrix(y_true=clf.Ytest, y_pred=clf.Ypred)
                        plot.confusion_matrix(cm)
                        result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif data_process == 'cross val score':
                        
                        clf.model = CatBoostClassifier(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'])
   
                        cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                        st.info('cv mean accuracy score: {}'.format(cvs.mean()))                 

        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')