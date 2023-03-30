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

st.write('## TUNE LOGISTICS REGRESSION PARAMETERs')
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
        
        colored_header(label="TUNE LOGISTICS REGRESSION PARAMETERs",description=" ",color_name="violet-70")

        # =================== model ====================================
        clf = CLASSIFIER(features,targets)

        target_selected_option = st.selectbox('target', list(clf.targets)[::-1])

        clf.targets = targets[target_selected_option]

        colored_header(label="CLASSIFIER", description=" ",color_name="violet-30")

        model_path = './models/tune parameters/'

        colored_header(label="TRAINING", description=" ",color_name="violet-30")

        # sss = st.slider('test',0,100,(25,75))

        template_alg = model_platform(model_path)

        inputs, col2,tab2 =  template_alg.show()
        
        if inputs['model'] == 'LogisticRegression':
            
            if inputs['tune_parameter'] == 'C':
                with col2:
                    with st.expander('Tune parameter'):
                        C = st.slider('C',1,1000,(5,50))
                        step = st.number_input('step',1,100,10)
                        cv = st.number_input('cv',1,10,5)
                        # scoring = st.selectbox('scoring',('r2', 'neg_mean_absolute_error','neg_mean_squared_error'))
                    with st.container():
                        button_train = st.button('train', use_container_width=True)
                    if button_train:
                        scorel = []

                        for i in range(C[0],C[1],step):

                            clf.model = LR(C=i, random_state=inputs['random state'])  
                            socre = cross_val_score(clf.model, clf.features, clf.targets, cv = cv).mean()
                            scorel.append(socre)

                        st.write(f'max socre {max(scorel)} in C = {[*range(C[0],C[1])][scorel.index(max(scorel))]}')
                        fig,ax = plt.subplots()
                        ax = plt.plot(range(C[0],C[1],step),scorel)
                        plt.xlabel('C')
                        plt.ylabel('accuracy')
                        st.pyplot(fig)
            elif inputs['tune_parameter'] == 'max iter':
                with col2:    
                    with st.expander('Tune parameter'):
                        max_iter = st.slider('max iter',1,1000,(1,20))
                        step = st.number_input('step',1,10,2)
                        cv = st.number_input('cv',1,10,5)
                        # scoring = st.selectbox('scoring',('r2','neg_mean_absolute_error','neg_mean_squared_error'))
                    with st.container():
                        button_train = st.button('train', use_container_width=True)
                    if button_train:
                        scorel = []
                        for i in range(max_iter[0],max_iter[1],step):
        
                            clf.model = LR(C=i, random_state=inputs['random state'])  
                            socre = cross_val_score(clf.model, clf.features, clf.targets, cv = cv).mean()
                            scorel.append(socre)
            
                            # my_bar.progress(times_j/((max_iter[1] - max_iter[0])%step), text=progress_text)
                        st.write(f'max socre {max(scorel)} in max iter = {[*range(max_iter[0],max_iter[1])][scorel.index(max(scorel))]}')
                        fig,ax = plt.subplots()
                        ax = plt.plot(range(max_iter[0],max_iter[1],step),scorel)
                        plt.xlabel('max iter')
                        plt.ylabel('accuracy')
                        st.pyplot(fig)           
        with tab2:
            colored_header(label="GRID SEARCH RFR", description=" ",color_name="blue-70")


        st.write('---')

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')