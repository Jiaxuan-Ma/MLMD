from utils import *
    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================


with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
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
    
        clf = CLASSIFIER(features,targets)

        target_selected_option = st.selectbox('target', list(clf.targets)[::-1])

        clf.targets = targets[target_selected_option]

        # st.write(fs.targets.head())
        colored_header(label="CLASSIFIER", description=" ",color_name="violet-30")

        model_path = './models/classifiers'
        
        template_alg = model_platform(model_path)

        colored_header(label="TRAINING", description=" ",color_name="violet-30")

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
                        scoring = st.selectbox('scoring',('accuracy','average_precision','f1','f1_micro','f1_macro','f1_weighted','f1_samples'))

            if st.button('train'):
                if data_process == 'train test split':
                            
                    clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                            max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],min_samples_split=inputs['min samples split'])
                                                               
                    clf.DecisionTreeClassifier()
                    
                    st.write('plot.........................')

                    plot = customPlot()
                    if inputs['tree graph']:
                        class_names = list(set(clf.targets.astype(str).tolist()))
                        dot_data = tree.export_graphviz(clf.model,out_file=None, feature_names=list(clf.features), class_names=class_names,filled=True, rounded=True)
                        graph = graphviz.Source(dot_data)
                        graph.render('Tree graph', view=True)
                elif data_process == 'cross val score':
                    clf.model = tree.DecisionTreeClassifier(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                                            max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],min_samples_split=inputs['min samples split'])
                                                            
                    cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv, scoring=scoring)

                    st.write('cross val score:', cvs)

        if inputs['model'] == 'RandomForestClassifier':
      
            with col2:
                with st.expander('Operator'):
                    data_process = st.selectbox('data process', ('train test split','cross val score','oob score'))
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif data_process == 'cross val score':
                        cv = st.number_input('cv',1,10,5)
                        scoring= st.selectbox('scoring',('accuracy','average_precision','f1','f1_micro','f1_macro','f1_weighted','f1_samples'))
                    elif data_process == 'oob score':
                        inputs['oob score']  = st.checkbox('oob score',True)
                        inputs['warm start'] = True
            with st.container():
                button_train = st.button('train', use_container_width=True)
            if button_train:

                if data_process == 'train test split':

                    clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'], warm_start=inputs['warm start'])
                    
                    clf.RandomForestClassifier()
                    st.write('plot......................')
                    plot = customPlot()

                elif data_process == 'cross val score':

 
                    clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'], warm_start=inputs['warm start'])
                
     
                    cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv,scoring=scoring)
     
                    st.write(f'mean cross val score: {cvs.mean()}')

                elif data_process == 'oob score':
 
                    clf.model = RFC(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'], oob_score=inputs['oob score'], warm_start=inputs['warm start'])
                
                    clf_res  = clf.model.fit(clf.features, clf.targets)
                    oob_score = clf_res.oob_score_
                    st.write(f'oob score : {oob_score}')              


        if inputs['model'] == 'LogisticRegression':

            with col2:
                with st.expander('Operator'):
                    data_process = st.selectbox('data process', ('train test split','cross val score'))
                    if data_process == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        clf.Xtrain, clf.Xtest, clf.Ytrain, clf.Ytest = TTS(clf.features,clf.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif data_process == 'cross val score':
                        cv = st.number_input('cv',1,10,5)
                        scoring = st.selectbox('scoring',('accuracy','average_precision','f1','f1_micro','f1_macro','f1_weighted','f1_samples'))
            with st.container():
                button_train = st.button('train', use_container_width=True)
            if button_train:
                if data_process == 'train test split':
                            
                    clf.model = LR(penalty=inputs['penalty'],C=inputs['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                   random_state=inputs['random state'],l1_ratio= inputs['l1 ratio'])   
                    clf.LogisticRegreesion()
                    
                    st.write('plot.........................')

                    plot = customPlot()
                elif data_process == 'cross val score':
                    clf.model = LR(penalty=inputs['penalty'],C=inputs['C'],solver=inputs['solver'],max_iter=inputs['max iter'],multi_class=inputs['multi class'],
                                   random_state=inputs['random state'],l1_ratio= inputs['l1 ratio'])    
                     
                    cvs = cross_val_score(clf.model, clf.features, clf.targets, cv = cv, scoring=scoring)

                    st.write('cross val score:', cvs)                

        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')