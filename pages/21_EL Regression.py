from utils import *
    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================

st.write('## ENSEMBLE REGRESSION')
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

        model_path = './models/el regressors'



        template_alg = model_platform(model_path)

        inputs, col2 = template_alg.show()

        if inputs['model'] == 'BaggingRegressor':
            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out'))
                   
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

                    if inputs['base estimator'] == "DecisionTree": 
                        reg.model = BaggingRegressor(estimator = tree.DecisionTreeRegressor(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                        max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                        
                        reg.BaggingRegressor()
        
                        plot = customPlot()

                        plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                    
                    elif inputs['base estimator'] == "SupportVector": 
                        reg.model = BaggingRegressor(estimator = SVR(),n_estimators=inputs['nestimators'],
                                max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                        reg.BaggingRegressor()
        
                        plot = customPlot()

                        plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)   
                    
                    elif inputs['base estimator'] == "LinearRegression": 
                        reg.model = BaggingRegressor(estimator = LinearR(),n_estimators=inputs['nestimators'],
                                max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                        reg.BaggingRegressor()
        
                        plot = customPlot()

                        plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)   

                elif operator == 'cross val score':
                    if inputs['base estimator'] == "DecisionTree": 
                        reg.model = BaggingRegressor(estimator = tree.DecisionTreeRegressor(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                                        max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                        cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)
            
                        st.write('mean cross val R2:', cvs.mean())
                    elif inputs['base estimator'] == "SupportVector": 

                        reg.model = BaggingRegressor(estimator =  SVR(),n_estimators=inputs['nestimators'],
                                max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                        cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)
            
                        st.write('mean cross val R2:', cvs.mean())  

                    elif inputs['base estimator'] == "LinearRegression":
                        reg.model = BaggingRegressor(estimator = LinearR(),n_estimators=inputs['nestimators'],
                                max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 

                        cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)
            
                        st.write('mean cross val R2:', cvs.mean())

                elif operator == 'leave one out':
                    if inputs['base estimator'] == "DecisionTree": 
                        reg.model = BaggingRegressor(estimator = tree.DecisionTreeRegressor(random_state=inputs['random state']),n_estimators=inputs['nestimators'],
                                max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                    
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
                    
                    elif inputs['base estimator'] == "SupportVector": 

                        reg.model = BaggingRegressor(estimator =  SVR(),n_estimators=inputs['nestimators'],
                                max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
                        
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

                    elif inputs['base estimator'] == "LinearRegression":
                        reg.model = BaggingRegressor(estimator = LinearR(),n_estimators=inputs['nestimators'],
                                max_samples=inputs['max samples'], max_features=inputs['max features'], n_jobs=-1) 
         
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

        if inputs['model'] == 'AdaBoostRegressor':

            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out'))
                   
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

                    if inputs['base estimator'] == "DecisionTree": 
                        reg.model = AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                        
                        reg.AdaBoostRegressor()
        
                        plot = customPlot()

                        plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                    
                    elif inputs['base estimator'] == "SupportVector": 

                        reg.model = AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
                        
                        reg.AdaBoostRegressor()
        
                        plot = customPlot()

                        plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)   
                    
                    elif inputs['base estimator'] == "LinearRegression": 
                        reg.model = AdaBoostRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state'])
                        reg.AdaBoostRegressor()
        
                        plot = customPlot()

                        plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)   

                elif operator == 'cross val score':
                    if inputs['base estimator'] == "DecisionTree": 
                        reg.model =  AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                        cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)
            
                        st.write('mean cross val R2:', cvs.mean())
                    elif inputs['base estimator'] == "SupportVector": 

                        reg.model =  AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 

                        cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)
            
                        st.write('mean cross val R2:', cvs.mean())  

                    elif inputs['base estimator'] == "LinearRegression":
                        reg.model =  AdaBoostRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 

                        cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)
            
                        st.write('mean cross val R2:', cvs.mean())

                elif operator == 'leave one out':
                    if inputs['base estimator'] == "DecisionTree": 
                        reg.model =  AdaBoostRegressor(estimator=tree.DecisionTreeRegressor(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                    
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
                    
                    elif inputs['base estimator'] == "SupportVector": 

                        reg.model = reg.model = AdaBoostRegressor(estimator=SVR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                        
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

                    elif inputs['base estimator'] == "LinearRegression":
                        reg.model = reg.model =  AdaBoostRegressor(estimator=LinearR(), n_estimators=inputs['nestimators'], learning_rate=inputs['learning rate'],random_state=inputs['random state']) 
                                              
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
                    
        if inputs['model'] == 'GradientBoostingRegressor':

            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out'))
                   
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

                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_depth=inputs['max depth'],max_features=inputs['max features'],
                                                               random_state=inputs['random state']) 
                        
                        reg.AdaBoostRegressor()
        
                        plot = customPlot()

                        plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':
                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_depth=inputs['max depth'],max_features=inputs['max features'],
                                                               random_state=inputs['random state'])  
                        cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)
            
                        st.write('mean cross val R2:', cvs.mean())

                elif operator == 'leave one out':
                        reg.model = GradientBoostingRegressor(learning_rate=inputs['learning rate'],n_estimators=inputs['nestimators'],max_depth=inputs['max depth'],max_features=inputs['max features'],
                                                               random_state=inputs['random state']) 
                    
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

        if inputs['model'] == 'XGBRegressor':

            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out'))
                   
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

                        reg.model = xgb.XGBRegressor(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                      max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                      learning_rate=inputs['learning rate'])
                        reg.XGBRegressor()
        
                        plot = customPlot()

                        plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':
                        reg.model = xgb.XGBRegressor(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                      max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                      learning_rate=inputs['learning rate'])
                        cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)
            
                        st.write('mean cross val R2:', cvs.mean())

                elif operator == 'leave one out':
                        reg.model = xgb.XGBRegressor(booster=inputs['base estimator'], n_estimators=inputs['nestimators'], 
                                                      max_depth= inputs['max depth'], subsample=inputs['subsample'], colsample_bytree=inputs['subfeature'], 
                                                      learning_rate=inputs['learning rate'])
                    
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


        if inputs['model'] == 'LGBMRegressor':

            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out'))
                   
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

                        reg.model = lgb.LGBMRegressor(niterations=inputs['niterations'],nestimators=inputs['nestimators'],learning_rate=inputs['learning rate'],
                                                    num_leaves=inputs['num_leaves'],max_depth=inputs['max depth'])

                        reg.LGBMRegressor()
        
                        plot = customPlot()

                        plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':
                        reg.model = lgb.LGBMRegressor(niterations=inputs['niterations'],nestimators=inputs['nestimators'],learning_rate=inputs['learning rate'],
                                                    num_leaves=inputs['num_leaves'],max_depth=inputs['max depth'])
                        cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)
            
                        st.write('mean cross val R2:', cvs.mean())

                elif operator == 'leave one out':
                        reg.model = lgb.LGBMRegressor(niterations=inputs['niterations'],nestimators=inputs['nestimators'],learning_rate=inputs['learning rate'],
                                                    num_leaves=inputs['num_leaves'],max_depth=inputs['max depth'])
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

        if inputs['model'] == 'CatBoostRegressor':
            with col2:
                with st.expander('Operator'):
                    operator = st.selectbox('data operator', ('train test split','cross val score','leave one out'))
                   
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

                        reg.model =CatBoostRegressor(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'])

                        reg.LGBMRegressor()
        
                        plot = customPlot()

                        plot.pred_vs_actual(reg.Ytest, reg.Ypred)

                        result_data = pd.concat([reg.Ytest, pd.DataFrame(reg.Ypred)], axis=1)
                        result_data.columns = ['actual','prediction']
                        with st.expander('Actual vs Predict'):
                            st.write(result_data)
                            tmp_download_link = download_button(result_data, f'prediction vs actual.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                elif operator == 'cross val score':
                        reg.model = CatBoostRegressor(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'])
                        cvs = CVS(reg.model, reg.features, reg.targets, cv = cv)
            
                        st.write('mean cross val R2:', cvs.mean())

                elif operator == 'leave one out':
                        reg.model = CatBoostRegressor(iterations=inputs['niteration'],learning_rate=inputs['learning rate'],depth = inputs['max depth'])
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
        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')