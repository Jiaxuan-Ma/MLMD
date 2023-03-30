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

st.write('## TUNE RANDOM FOREST PARAMETERs')
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
        
        colored_header(label="TUNE RANDOM FOREST PARAMETERs",description=" ",color_name="violet-70")

        # =================== model ====================================
        reg = REGRESSOR(features,targets)

        target_selected_option = st.selectbox('target', list(reg.targets)[::-1])

        reg.targets = targets[target_selected_option]

        colored_header(label="RREGRESSOR", description=" ",color_name="violet-30")

        model_path = './models/tune parameters/'

        colored_header(label="TRAINING", description=" ",color_name="violet-30")

        # sss = st.slider('test',0,100,(25,75))

        template_alg = model_platform(model_path)

        inputs, col2,tab2 =  template_alg.show()
        
        if inputs['model'] == 'RandomForestRegressor':
            
            if inputs['tune_parameter'] == 'nestimators':
                with col2:
                    with st.expander('Tune parameter'):
                        nestimators = st.slider('nestimators',1,1000,(5,50))
                        step = st.number_input('step',1,100,10)

                        cv = st.number_input('cv',1,10,5)
                        scoring = st.selectbox('scoring',('r2', 'neg_mean_absolute_error','neg_mean_squared_error'))
                    with st.container():
                        button_train = st.button('train', use_container_width=True)
                    if button_train:
                        scorel = []

                        for i in range(nestimators[0],nestimators[1],step):
                            reg.model = RFR(criterion = inputs['criterion'], n_estimators=i+1,
                                            random_state=inputs['random state'], n_jobs=inputs['njobs'])
                            socre = cross_val_score(reg.model, reg.features, reg.targets, cv = cv, scoring=scoring).mean()
                            scorel.append(socre)
            

                        st.write(f'max socre {max(scorel)} in nestimator = {[*range(nestimators[0],nestimators[1])][scorel.index(max(scorel))]}')
                        fig,ax = plt.subplots()
                        ax = plt.plot(range(nestimators[0],nestimators[1],step),scorel)
                        plt.xlabel('nestimator')
                        plt.ylabel(scoring)
                        st.pyplot(fig)

            elif inputs['tune_parameter'] == 'max depth':

                with col2:    
                    with st.expander('Tune parameter'):
                        max_depth = st.slider('max depth',1,1000,(1,20))
                        step = st.number_input('step',1,10,2)
                        cv = st.number_input('cv',1,10,5)
                        scoring = st.selectbox('scoring',('r2','neg_mean_absolute_error','neg_mean_squared_error'))
                    if st.button('train'):
                        scorel = []
                        for i in range(max_depth[0],max_depth[1],step):
         
                            reg.model = RFR(criterion = inputs['criterion'], max_depth=i+1,
                                            random_state=inputs['random state'], n_jobs=inputs['njobs'])
                            socre = cross_val_score(reg.model, reg.features, reg.targets, cv = cv, scoring=scoring).mean()
                            scorel.append(socre)
            
                            # my_bar.progress(times_j/((max_depth[1] - max_depth[0])%step), text=progress_text)
                        st.write(f'max socre {max(scorel)} in nestimator = {[*range(max_depth[0],max_depth[1])][scorel.index(max(scorel))]}')
                        fig,ax = plt.subplots()
                        ax = plt.plot(range(max_depth[0],max_depth[1],step),scorel)
                        plt.xlabel('max depth')
                        plt.ylabel(scoring)
                        st.pyplot(fig)
        with tab2:
            colored_header(label="GRID SEARCH RFR", description=" ",color_name="blue-70")

            g_nestimators = st.checkbox('gnestimators',True)

            if g_nestimators:
            
                g_n_estimators =st.slider('nestimators',1, 1000,(2,20))
                
                g_n_estimators_step = st.number_input('nestimator step',1,10,2)
            else:
                g_n_estimators = (100,101)
                g_n_estimators_step = 1

            g_maxdepth = st.checkbox('gmax depth',False)
            if g_maxdepth:
                g_max_depth = st.slider('max depth',1,1000,(1,200))
                g_maxdepth_step = st.number_input('max depth step',1,10,2)
            else:
                g_max_depth = [5,6]
                g_maxdepth_step = 1

            g_maxleafnodes = st.checkbox('gmax leaf nodes',False)
            if g_maxleafnodes:
                g_max_leaf_nodes = st.slider('max leaf nodes',20,1000,(20,200))
                g_maxleafnodes_step = st.number_input('max leaf nodes step',1,10,2)
            else:
                g_max_leaf_nodes = [20,21]
                g_maxleafnodes_step = 1

            g_minsamplesleaf = st.checkbox('gmin samples leaf',False)
            if g_minsamplesleaf:
                g_min_samples_leaf = st.slider('min samples leaf',1,200,(1,20))
                g_minsamplesleaf_step = st.number_input('min samples leaf step',1,50,2)
            else:
                g_min_samples_leaf = [1,2]
                g_minsamplesleaf_step = 1

            g_minsamplessplit = st.checkbox('gmin samples split', False)
            if g_minsamplessplit:
                g_min_samples_split = st.slider('min samples split',2,200,(1,20))
                g_minsamplessplit_step = st.number_input('min samples split step',1,50,2)
            else:
                g_min_samples_split = [2,3]
                g_minsamplessplit_step = 1

            g_criterion_two = st.checkbox('gcriterion',False)   
            if g_criterion_two:
                g_criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
            else: 
                g_criterion = ['squared_error']

            g_maxfeatures = st.checkbox('gmax features', False)
            if g_maxfeatures:
                g_max_features = st.slider('max features',2,100,(2,20))
                g_maxfeatures_step = st.number_input('max features step',1,10,2)
            else:
                g_max_features = [1,2]
                g_maxfeatures_step = 1
            
            cv = st.number_input('cv',1,10,5, key=12345)
            scoring = st.selectbox('scoring',('r2','neg_mean_absolute_error','neg_mean_squared_error'), key=43211)

            param_grid = {'max_depth': np.arange(g_max_depth[0],g_max_depth[1],g_maxdepth_step),
                          'max_leaf_nodes': np.arange(g_max_leaf_nodes[0],g_max_leaf_nodes[1],g_maxleafnodes_step),
                          'criterion':g_criterion,
                          'min_samples_leaf':np.arange(g_min_samples_leaf[0],g_min_samples_leaf[1],g_minsamplesleaf_step),
                          'min_samples_split':np.arange(g_min_samples_split[0],g_min_samples_split[1], g_minsamplessplit_step),
                          'max_features':np.arange(g_max_features[0],g_max_features[1],g_maxfeatures_step),
                          'n_estimators': np.arange(g_n_estimators[0],g_n_estimators[1], g_n_estimators_step)}
            reg.model = RFR(random_state=1024)
            GS = GridSearchCV(reg.model, param_grid, cv = cv)
            with st.container():
                button_train = st.button('train', use_container_width=True, key=4994)
            if button_train:
                GS.fit(reg.features, reg.targets)
                st.write(GS.best_params_)
                st.write(f'Grid search best score: {GS.best_score_}')

        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
