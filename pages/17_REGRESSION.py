from utils import *
    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================

st.write('## REGRESSION')
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
            # =================== check =============================
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
    # =================== model ====================================
        reg = REGRESSOR(features,targets)

        target_selected_option = st.selectbox('target', list(reg.targets)[::-1])

        reg.targets = targets[target_selected_option]

        colored_header(label="RREGRESSOR", description=" ",color_name="violet-30")

        model_path = './models/regressors'

        colored_header(label="TRAINING", description=" ",color_name="violet-30")

        template_alg = model_platform(model_path)

        inputs, col2 = template_alg.show()
    
        if inputs['model'] == 'DecisionTreeRegressor':

            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score'))
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])

                    elif operator == 'cross val score':
                        cv = st.number_input('cv',1,10,5)
                        scoring = st.selectbox('scoring',('r2','neg_mean_absolute_error','neg_mean_squared_error'))

            with st.container():
                button_train = st.button('train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = tree.DecisionTreeRegressor(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                                    max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                    min_samples_split=inputs['min samples split']) 
                    
                    reg.DecisionTreeRegressor()
                    plot = customPlot()
                    plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    with st.expander('ACTUAL AND PREDICTION DATA'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                    if inputs['tree graph']:
                        class_names = list(set(reg.targets.astype(str).tolist()))
                        dot_data = tree.export_graphviz(reg.model,out_file=None, feature_names=list(reg.features), class_names=class_names,filled=True, rounded=True)
                        graph = graphviz.Source(dot_data)
                        graph.render('Tree graph', view=True)

                elif operator == 'cross val score':

                    reg.model = tree.DecisionTreeRegressor(criterion=inputs['criterion'],random_state=inputs['random state'],splitter=inputs['splitter'],
                        max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                        min_samples_split=inputs['min samples split']) 

                    cvs = CVS(reg.model, reg.features, reg.targets, cv = cv, scoring=scoring)

                    st.write('mean cross val score:', cvs.mean())

        if inputs['model'] == 'RandomForestRrgressor':
            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','oob score'))
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif operator == 'cross val score':
                        cv = st.number_input('cv',1,10,5)
                        scoring = st.selectbox('scoring',('r2','neg_mean_absolute_error','neg_mean_squared_error'))
                    elif operator == 'oob score':
                        inputs['oob score']  = st.selectbox('oob score',[True], disabled=True)
                        inputs['warm start'] = True

            with st.container():
                button_train = st.button('train', use_container_width=True)
            if button_train:

                if operator == 'train test split':

                    reg.model = RFR(criterion = inputs['criterion'], n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                    min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                    n_jobs=inputs['njobs'])
                    
                    reg.RandomForestRegressor()
    
                    plot = customPlot()

                    plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    with st.expander('ACTUAL AND PREDICTION DATA'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':

                    reg.model = RFR(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                n_jobs=inputs['njobs'])

                    cvs = CVS(reg.model, reg.features, reg.targets, cv = cv, scoring=scoring, n_jobs=inputs['njobs'])
        
                    st.write('mean cross val score:', cvs.mean())
                    st.write('mean cross val score:', cvs)
                    # st.write(inputs['criterion'])
                    # st.write(scoring)

                elif operator == 'oob score':
                    # st.write(inputs['warm start'])
                    reg.model = RFR(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                n_jobs=inputs['njobs'])
                
                    reg_res  = reg.model.fit(reg.features, reg.targets)
                    oob_score = reg_res.oob_score_
                    st.write(f'oob score : {oob_score}')
        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')