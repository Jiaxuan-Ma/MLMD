from utils import *
    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# ======================================================

st.write('## CLASSIFICATION')
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

        model_path = './models/classifiers'
        
        template_alg = model_platform(model_path)

        colored_header(label="Training", description=" ",color_name="violet-30")

        inputs, col2 = template_alg.show()
       
        if inputs['model'] == 'DecisionTreeClassifier':

            with col2:
                with st.expander('Operator'):
                    data_process = st.selectbox('data process', ('train test split','cross val score'))
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif data_process == 'cross val score':
                        cv = st.number_input('cv',1,10,5)
                 

            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if data_process == 'train test split':
                            
                    clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                            max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],min_samples_split=inputs['min samples split'])
                                                               
                    clf.DecisionTreeClassifier()

                    plot = customPlot()
                    cm = confusion_matrix(y_true=clf.Ytest, y_pred=clf.Ypred)
                    plot.confusion_matrix(cm)
                    result_data = pd.concat([clf.Ytest, pd.DataFrame(clf.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)
                    if inputs['tree graph']:
                        class_names = list(set(clf.targets.astype(str).tolist()))
                        dot_data = tree.export_graphviz(clf.model,out_file=None, feature_names=list(clf.features), class_names=class_names,filled=True, rounded=True)
                        graph = graphviz.Source(dot_data)
                        graph.render('Tree graph', view=True)
                elif data_process == 'cross val score':
                    clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                            max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],min_samples_split=inputs['min samples split'])
                                                            
                    cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                    st.write('cv mean accuracy score: {}'.format(cvs.mean())) 

        if inputs['model'] == 'RandomForestClassifier':
      
            with col2:
                with st.expander('Operator'):
                    data_process = st.selectbox('data process', ('train test split','cross val score','oob score'))
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif data_process == 'cross val score':
                        cv = st.number_input('cv',1,10,5)
             
                    elif data_process == 'oob score':
                        inputs['oob score']  = st.checkbox('oob score',True)
                        inputs['warm start'] = True
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:

                if data_process == 'train test split':

                    clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'], warm_start=inputs['warm start'])
                    
                    clf.RandomForestClassifier()
                    
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

 
                    clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'], warm_start=inputs['warm start'])
                
     
                    cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)
     
                    st.write('cv mean accuracy score: {}'.format(cvs.mean())) 

                elif data_process == 'oob score':
 
                    clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'], warm_start=inputs['warm start'])
                
                    clf_res  = clf.model.fit(clf.features, clf.targets)
                    oob_score = clf_res.oob_score_
                    st.write(f'oob score : {oob_score}')              


        if inputs['model'] == 'LogisticRegression':

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
                            
                    clf.model = LR(penalty=inputs['penalty'],C=inputs['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                   random_state=inputs['random state'],l1_ratio= inputs['l1 ratio'])   
                    clf.LogisticRegreesion()
                    
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
                    clf.model = LR(penalty=inputs['penalty'],C=inputs['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                   random_state=inputs['random state'],l1_ratio= inputs['l1 ratio'])   
                     
                    cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)

                    st.write('cv mean accuracy score: {}'.format(cvs.mean())) 

        if inputs['model'] == 'SupportVector':
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
                            
                    clf.model = SVC(C=inputs['C'], kernel=inputs['kernel'], class_weight=inputs['class weight'])
                                                               
                    clf.SupportVector()
                    
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
                    clf.model = SVC(C=inputs['C'], kernel=inputs['kernel'], class_weight=inputs['class weight'])
                                                                                             
                    cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv)
                    st.write('cv mean accuracy score: {}'.format(cvs.mean()))     

        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')