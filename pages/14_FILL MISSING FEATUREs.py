from utils import *


    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================

st.write('## FILL MISSING FEATUREs')
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

            colored_header(label="TABLE", description=" ",color_name="blue-70")
            nrow = st.slider("rows", 1, len(df)-1, 5)
            df_nrow = df.head(nrow)
            st.write(df_nrow)

            colored_header(label="FEATUREs vs TARGET",description=" ",color_name="blue-30")

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
        
        #=============== drop major missing features ================
    
        colored_header(label="FILL MISSING FEATUREs",description=" ",color_name="violet-70")
        fs = FeatureSelector(features, targets)
        missing_feature_list = fs.features.columns[fs.features.isnull().any()].tolist()
        # assert len(fs.features.columns[fs.features.isnull().any()].tolist()) == 0,'Zero missing feature!'

        with st.container():
            fill_method = st.selectbox('fill method',('fill in Normal method', 'fill in RandomFrostRegression'))
        
        if fill_method == 'fill in Normal method':
            if len(missing_feature_list) == 0:
                st.error('Zero missing feature!')
                st.stop()
            missing_feature = st.multiselect('missing feature',missing_feature_list,missing_feature_list[-1])
            
            option_filled = st.selectbox('mean',('mean','constant','median','most frequent'))
            if option_filled == 'mean':
                # fs.features[missing_feature] = fs.features[missing_feature].fillna(fs.features[missing_feature].mean())
                imp = SimpleImputer(missing_values=np.nan,strategy= 'mean')

                fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])
            elif option_filled == 'constant':
                # fs.features[missing_feature] = fs.features[missing_feature].fillna(0)
                fill_value = st.number_input('Insert a constant')
                imp = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value = fill_value)
                
                fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])
            elif option_filled == 'median':
                # fs.features[missing_feature] = fs.features[missing_feature].fillna(0)
                imp = SimpleImputer(missing_values=np.nan, strategy= 'median')
                
                fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])
            elif option_filled == 'most frequent':

                imp = SimpleImputer(missing_values=np.nan, strategy= 'most_frequent')
                
                fs.features[missing_feature] = imp.fit_transform(fs.features[missing_feature])  

            data = pd.concat([fs.features, targets], axis=1)

        else:
            with st.expander('Hyper Parameters'):
                num_estimators = st.number_input('number estimators',1, 10000, 100)
                criterion = st.selectbox('criterion',('squared_error','absolute_error','friedman_mse','poisson'))
                max_depth = st.number_input('max depth',1, 1000, 5)
                min_samples_leaf = st.number_input('min samples leaf', 1, 1000, 5)
                min_samples_split = st.number_input('min samples split', 1, 1000, 5)
                random_state = st.checkbox('random state 1024',True)


            option_filled = st.selectbox('mean',('mean','constant','median','most frequent'))
            if option_filled == 'mean':
                feature_missing_reg = fs.features.copy()
                
                null_columns = feature_missing_reg.columns[feature_missing_reg.isnull().any()] 
       
                null_counts = feature_missing_reg.isnull().sum()[null_columns].sort_values() 
                null_columns_ordered = null_counts.index.tolist() 
  
                for i in null_columns_ordered:
 
                    df = feature_missing_reg
                    fillc = df[i]
     
                    df = pd.concat([df.iloc[:,df.columns != i], pd.DataFrame(targets)], axis=1)
             
                    df_temp_fill = SimpleImputer(missing_values=np.nan,strategy= 'mean').fit_transform(df)

                    YTrain = fillc[fillc.notnull()]
                    YTest = fillc[fillc.isnull()]
                    XTrain = df_temp_fill[YTrain.index,:]
                    XTest = df_temp_fill[YTest.index,:]

                    rfc = RFR(n_estimators=num_estimators,criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                              min_samples_split=min_samples_split,random_state=random_state)

                    rfc = rfc.fit(XTrain, YTrain)
                    YPredict = rfc.predict(XTest)
           
                   
                    feature_missing_reg.loc[feature_missing_reg[i].isnull(), i] = YPredict

            elif option_filled == 'constant':

                fill_value = st.number_input('Insert a constant')
                feature_missing_reg = fs.features.copy()
                
                null_columns = feature_missing_reg.columns[feature_missing_reg.isnull().any()] 
       
                null_counts = feature_missing_reg.isnull().sum()[null_columns].sort_values() 
                null_columns_ordered = null_counts.index.tolist() 
  
                for i in null_columns_ordered:

                    df = feature_missing_reg
                    fillc = df[i]
     
                    df = pd.concat([df.iloc[:,df.columns != i], pd.DataFrame(targets)], axis=1)

                    df_temp_fill = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value = fill_value).fit_transform(df)

                    YTrain = fillc[fillc.notnull()]
                    YTest = fillc[fillc.isnull()]
                    XTrain = df_temp_fill[YTrain.index,:]
                    XTest = df_temp_fill[YTest.index,:]

                    rfc = RFR(n_estimators=num_estimators,criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                              min_samples_split=min_samples_split,random_state=random_state)

                    rfc = rfc.fit(XTrain, YTrain)
                    YPredict = rfc.predict(XTest)

                    feature_missing_reg.loc[feature_missing_reg[i].isnull(), i] = YPredict
                
            elif option_filled == 'median':
                feature_missing_reg = fs.features.copy()
            
                null_columns = feature_missing_reg.columns[feature_missing_reg.isnull().any()] 
        
                null_counts = feature_missing_reg.isnull().sum()[null_columns].sort_values() 
                null_columns_ordered = null_counts.index.tolist() 

                for i in null_columns_ordered:
   
                    df = feature_missing_reg
                    fillc = df[i]
        
                    df = pd.concat([df.iloc[:,df.columns != i], pd.DataFrame(targets)], axis=1)
   
                    df_temp_fill = SimpleImputer(missing_values=np.nan,strategy= 'median').fit_transform(df)

                    YTrain = fillc[fillc.notnull()]
                    YTest = fillc[fillc.isnull()]
                    XTrain = df_temp_fill[YTrain.index,:]
                    XTest = df_temp_fill[YTest.index,:]

                    rfc = RFR(n_estimators=num_estimators,criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                              min_samples_split=min_samples_split,random_state=random_state)

                    rfc = rfc.fit(XTrain, YTrain)
                    YPredict = rfc.predict(XTest)
  
                    feature_missing_reg.loc[feature_missing_reg[i].isnull(), i] = YPredict
     
            elif option_filled == 'most frequent':

                feature_missing_reg = fs.features.copy()
            
                null_columns = feature_missing_reg.columns[feature_missing_reg.isnull().any()] 
        
                null_counts = feature_missing_reg.isnull().sum()[null_columns].sort_values() 
                null_columns_ordered = null_counts.index.tolist() 

                for i in null_columns_ordered:

                    df = feature_missing_reg
                    fillc = df[i]
        
                    df = pd.concat([df.iloc[:,df.columns != i], pd.DataFrame(targets)], axis=1)
      
                    df_temp_fill = SimpleImputer(missing_values=np.nan,strategy= 'most_frequent').fit_transform(df)

                    YTrain = fillc[fillc.notnull()]
                    YTest = fillc[fillc.isnull()]
                    XTrain = df_temp_fill[YTrain.index,:]
                    XTest = df_temp_fill[YTest.index,:]

                    rfc = RFR(n_estimators=num_estimators,criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                              min_samples_split=min_samples_split,random_state=random_state)
                    rfc = rfc.fit(XTrain, YTrain)
                    YPredict = rfc.predict(XTest)
                    
                    feature_missing_reg.loc[feature_missing_reg[i].isnull(), i] = YPredict

            data = pd.concat([feature_missing_reg, targets], axis=1)

            
        st.write(data)

        tmp_download_link = download_button(data, f'fill missing features.csv', button_text='download')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


