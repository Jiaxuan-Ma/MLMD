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
        colored_header(label="Data Information", description=" ", color_name="violet-70")
        with st.expander('Data Information'):
            df = pd.read_csv(file)
            # =================== check =============================
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
    # =================== model ====================================
        reg = REGRESSOR(features,targets)

        colored_header(label="Choose Target", description=" ", color_name="violet-30")

        target_selected_option = st.selectbox('target', list(reg.targets)[::-1])

        reg.targets = targets[target_selected_option]

        colored_header(label="Regressor", description=" ",color_name="violet-30")

        model_path = './models/regressors'


        template_alg = model_platform(model_path)

        inputs, col2 = template_alg.show()
    
        if inputs['model'] == 'DecisionTreeRegressor':

            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score', 'leave one out'))
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])

                    elif operator == 'cross val score':
                        cv = st.number_input('cv',1,10,5)
                    
                    elif operator == 'leave one out':
                        loo = LeaveOneOut()
            

            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],splitter=inputs['splitter'],
                                    max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                    min_samples_split=inputs['min samples split']) 
                    
                    reg.DecisionTreeRegressor()
                    plot = customPlot()
                    plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                    if inputs['tree graph']:
                        class_names = list(set(reg.targets.astype(str).tolist()))
                        dot_data = tree.export_graphviz(reg.model,out_file=None, feature_names=list(reg.features), class_names=class_names,filled=True, rounded=True)
                        graph = graphviz.Source(dot_data)
                        graph.render('Tree graph', view=True)

                elif operator == 'cross val score':

                    reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],splitter=inputs['splitter'],
                        max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                        min_samples_split=inputs['min samples split']) 

                    cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)

                    st.write('mean cross val R2:', cvs.mean())

                elif operator == 'leave one out':

                    reg.model = tree.DecisionTreeRegressor(random_state=inputs['random state'],splitter=inputs['splitter'],
                        max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                        min_samples_split=inputs['min samples split']) 
                   
                    Y_pred  =[]
                    Y_test = []
                    features = reg.features.values
                    targets = reg.targets.values
                    for train,test in loo.split(features):
                        Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
                        
                        reg.model.fit(Xtrain, Ytrain)
                        Ypred = reg.model.predict(Xtest)
                        Y_pred.append(Ypred)
                        Y_test.append(Ytest)

                    score = r2_score(y_true=Y_test,y_pred=Y_pred)
          
                    plot = customPlot()
                    plot.pred_vs_actual(Y_test, Y_pred)                  
                    st.write('mean cross val R2:', score)


        if inputs['model'] == 'RandomForestRegressor':
            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out','oob score'))
                   
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                    elif operator == 'cross val score':
                        cv = st.number_input('cv',1,10,5)

                    elif operator == 'leave one out':
                        loo = LeaveOneOut()

                    elif operator == 'oob score':
                        inputs['oob score']  = st.selectbox('oob score',[True], disabled=True)
                        inputs['warm start'] = True
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:

                if operator == 'train test split':

                    reg.model = RFR( n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                    min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                    n_jobs=inputs['njobs'])
                    
                    reg.RandomForestRegressor()
    
                    plot = customPlot()

                    plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':

                    reg.model = RFR(n_estimators=inputs['nestimators'],random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                n_jobs=inputs['njobs'])

                    cvs = CVS(reg.model, reg.features, reg.targets, cv = cv, n_jobs=inputs['njobs'])
        
                    st.write('mean cross val R2:', cvs.mean())
     


                elif operator == 'oob score':

                    reg.model = RFR(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                n_jobs=inputs['njobs'])
                
                    reg_res  = reg.model.fit(reg.features, reg.targets)
                    oob_score = reg_res.oob_score_
                    st.write(f'oob score : {oob_score}')

                elif operator == 'leave one out':

                    reg.model = RFR(criterion = inputs['criterion'],n_estimators=inputs['nestimators'] ,random_state=inputs['random state'],max_depth=inputs['max depth'],min_samples_leaf=inputs['min samples leaf'],
                                                min_samples_split=inputs['min samples split'],oob_score=inputs['oob score'], warm_start=inputs['warm start'],
                                                n_jobs=inputs['njobs'])
                   
                    Y_pred  =[]
                    Y_test = []
                    features = reg.features.values
                    targets = reg.targets.values
                    for train,test in loo.split(features):
                        Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
                        
                        reg.model.fit(Xtrain, Ytrain)
                        Ypred = reg.model.predict(Xtest)
                        Y_pred.append(Ypred)
                        Y_test.append(Ytest)

                    score = r2_score(y_true=Y_test,y_pred=Y_pred)
          
                    plot = customPlot()
                    plot.pred_vs_actual(Y_test, Y_pred)                  
                    st.write('mean cross val R2:', score)

        if inputs['model'] == 'SupportVector':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('data operator', ('train test split','cross val score', 'leave one out'))
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])

                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,10,5)

                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])
                    
                    reg.SupportVector()
                    plot = customPlot()
                    plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':

                    reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])

                    cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)

                    st.write('mean cross val R2:', cvs.mean())

                elif operator == 'leave one out':

                    reg.model = SVR(kernel=inputs['kernel'], C=inputs['C'])
                   
                    Y_pred  =[]
                    Y_test = []
                    features = pd.DataFrame(reg.features).values
                    targets = reg.targets.values
                    for train,test in loo.split(features):
                        Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
                        
                        reg.model.fit(Xtrain, Ytrain)
                        Ypred = reg.model.predict(Xtest)
                        Y_pred.append(Ypred)
                        Y_test.append(Ytest)

                    score = r2_score(y_true=Y_test,y_pred=Y_pred)
          
                    plot = customPlot()
                    plot.pred_vs_actual(Y_test, Y_pred)                  
                    st.write('mean cross val R2:', score)
                    
        if inputs['model'] == 'KNeighborsRegressor':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('data operator', ('train test split','cross val score', 'leave one out'))
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,10,5)

                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = KNeighborsRegressor(n_neighbors = inputs['n neighbors'])
                    
                    reg.KNeighborsRegressor()
                    plot = customPlot()
                    plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':

                    reg.model = KNeighborsRegressor(n_neighbors = inputs['n neighbors'])

                    cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)

                    st.write('mean cross val R2:', cvs.mean())

                elif operator == 'leave one out':

                    reg.model = KNeighborsRegressor(n_neighbors = inputs['n neighbors'])
                   
                    Y_pred  =[]
                    Y_test = []

                    features = pd.DataFrame(reg.features).values
                    targets = reg.targets.values
                    for train,test in loo.split(features):
                        Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
                        
                        reg.model.fit(Xtrain, Ytrain)
                        Ypred = reg.model.predict(Xtest)
                        Y_pred.append(Ypred)
                        Y_test.append(Ytest)

                    score = r2_score(y_true=Y_test,y_pred=Y_pred)
          
                    plot = customPlot()
                    plot.pred_vs_actual(Y_test, Y_pred)                  
                    st.write('mean cross val R2:', score)

        if inputs['model'] == 'LinearRegressor':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('data operator', ('train test split','cross val score', 'leave one out'))
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,10,5)
     
                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = LinearR()
                    
                    reg.LinearRegressor()
                    plot = customPlot()
                    plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':

                    reg.model = LinearR()

                    cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)

                    st.write('mean cross val score:', cvs.mean())

                elif operator == 'leave one out':

                    reg.model = LinearR()
                   
                    Y_pred  =[]
                    Y_test = []
                    features = pd.DataFrame(reg.features).values
                    targets = reg.targets.values
                    for train,test in loo.split(features):
                        Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
                        
                        reg.model.fit(Xtrain, Ytrain)
                        Ypred = reg.model.predict(Xtest)
                        Y_pred.append(Ypred)
                        Y_test.append(Ytest)

                    score = r2_score(y_true=Y_test,y_pred=Y_pred)
          
                    plot = customPlot()
                    plot.pred_vs_actual(Y_test, Y_pred)                  
                    st.write('mean cross val R2:', score)
        
        if inputs['model'] == 'LassoRegressor':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('data operator', ('train test split','cross val score', 'leave one out'))
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,10,5)

                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()
               
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])
                    
                    reg.LassoRegressor()
                    plot = customPlot()
                    plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':

                    reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])

                    cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)

                    st.write('mean cross val score:', cvs.mean())

                elif operator == 'leave one out':

                    reg.model = Lasso(alpha=inputs['alpha'],random_state=inputs['random state'])
                   
                    Y_pred  =[]
                    Y_test = []
                    features = pd.DataFrame(reg.features).values
                    targets = reg.targets.values
                    for train,test in loo.split(features):
                        Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
                        
                        reg.model.fit(Xtrain, Ytrain)
                        Ypred = reg.model.predict(Xtest)
                        Y_pred.append(Ypred)
                        Y_test.append(Ytest)

                    score = r2_score(y_true=Y_test,y_pred=Y_pred)
          
                    plot = customPlot()
                    plot.pred_vs_actual(Y_test, Y_pred)                  
                    st.write('mean cross val R2:', score)

        if inputs['model'] == 'RidgeRegressor':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('data operator', ('train test split','cross val score', 'leave one out'))
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,10,5)
                    
                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()              
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])
                    
                    reg.RidgeRegressor()
                    plot = customPlot()
                    plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':

                    reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])

                    cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)

                    st.write('mean cross val score:', cvs.mean())

                elif operator == 'leave one out':

                    reg.model = Ridge(alpha=inputs['alpha'], random_state=inputs['random state'])
                   
                    Y_pred  =[]
                    Y_test = []
                    features = pd.DataFrame(reg.features).values
                    targets = reg.targets.values
                    for train,test in loo.split(features):
                        Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
                        
                        reg.model.fit(Xtrain, Ytrain)
                        Ypred = reg.model.predict(Xtest)
                        Y_pred.append(Ypred)
                        Y_test.append(Ytest)

                    score = r2_score(y_true=Y_test,y_pred=Y_pred)
          
                    plot = customPlot()
                    plot.pred_vs_actual(Y_test, Y_pred)                  
                    st.write('mean cross val R2:', score)


        if inputs['model'] == 'MLPRegressor':

            with col2:
                with st.expander('Operator'):

                    preprocess = st.selectbox('data preprocess',['StandardScaler','MinMaxScaler'])

                    operator = st.selectbox('data operator', ('train test split','cross val score', 'leave one out'))
                    if operator == 'train test split':
                        inputs['test size'] = st.slider('test size',0.1, 0.5, 0.2)  
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        
                        reg.features = pd.DataFrame(reg.features)    
                        
                        reg.Xtrain, reg.Xtest, reg.Ytrain, reg.Ytest = TTS(reg.features,reg.targets,test_size=inputs['test size'],random_state=inputs['random state'])
                        
                    elif operator == 'cross val score':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        cv = st.number_input('cv',1,10,5)
                    
                    elif operator == 'leave one out':
                        if preprocess == 'StandardScaler':
                            reg.features = StandardScaler().fit_transform(reg.features)
                        if preprocess == 'MinMaxScaler':
                            reg.features = MinMaxScaler().fit_transform(reg.features)
                        loo = LeaveOneOut()              
            colored_header(label="Training", description=" ",color_name="violet-30")
            with st.container():
                button_train = st.button('Train', use_container_width=True)
            if button_train:
                if operator == 'train test split':

                    reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'], 
                                             batch_size=inputs['batch size'], learning_rate= inputs['learning rate'], max_iter=inputs['max iter'],
                                             random_state=inputs['random state'])
                    reg.MLPRegressor()
                    plot = customPlot()
                    plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                    result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                    result_data.columns = ['actual','prediction']
                    
                    with st.expander('Actual vs Predict'):
                        st.write(result_data)
                        tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':

                    reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'], 
                                             batch_size=inputs['batch size'], learning_rate= inputs['learning rate'], max_iter=inputs['max iter'],
                                             random_state=inputs['random state'])
                    
                    cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)

                    st.write('mean cross val score:', cvs.mean())

                elif operator == 'leave one out':

                    reg.model = MLPRegressor(hidden_layer_sizes=inputs['hidden layer size'], activation= inputs['activation'], solver=inputs['solver'], 
                                             batch_size=inputs['batch size'], learning_rate= inputs['learning rate'], max_iter=inputs['max iter'],
                                             random_state=inputs['random state'])
                   
                    Y_pred  =[]
                    Y_test = []
                    features = pd.DataFrame(reg.features).values
                    targets = reg.targets.values
                    for train,test in loo.split(features):
                        Xtrain, Xtest, Ytrain,Ytest = features[train],features[test],targets[train],targets[test]
                        
                        reg.model.fit(Xtrain, Ytrain)
                        Ypred = reg.model.predict(Xtest)
                        Y_pred.append(Ypred)
                        Y_test.append(Ytest)

                    score = r2_score(y_true=Y_test,y_pred=Y_pred)
          
                    plot = customPlot()
                    plot.pred_vs_actual(Y_test, Y_pred)                  
                    st.write('mean cross val R2:', score)   

        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')