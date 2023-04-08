
from utils import *


    # streamlit_analytics.start_tracking()

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================================================

with st.sidebar:
    badge(type="buymeacoffee", name="jiaxuanmasw")
# ======================================================

st.write('## Active learning')
st.write('---')

# =====================================================

if st.session_state["authentication_status"]:

    file = st.file_uploader("Upload `.csv`file", type=['csv'], label_visibility="collapsed", accept_multiple_files=True)
    upload_file = option_menu(None, ["Upload"], icons=['cloud-upload'], menu_icon="cast", 
                              default_index=0, orientation="horizontal",styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "25px"}, 
                "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "gray"}})  
    if len(file) > 2:
        st.error('Only upload two files, the first is the data set, the second is the the vritual space sample point.')
        st.stop()
    if len(file) == 2:
        st.warning('You have unpload two files, the first is the dataset, the second is the the vritual space sample point.')       
    if len(file) > 0:

        colored_header(label="Data Information",description=" ",color_name="violet-70")

        with st.expander('Data Information'):
            df = pd.read_csv(file[0])
            if len(file) == 2:
                df_vs = pd.read_csv(file[1])
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

        # colored_header(label="Active learning", description=" ", color_name="violet-70")

        sp = SAMPLING(features, targets)

        colored_header(label="Choose Target", description=" ", color_name="violet-30")

        target_selected_option = st.selectbox('target', list(sp.targets))
        
        sp.targets = sp.targets[target_selected_option]
        

        colored_header(label="Sampling", description=" ",color_name="violet-30")

        model_path = './models/active learning'

        colored_header(label="Training", description=" ",color_name="violet-30")

        template_alg = model_platform(model_path)

        inputs, col2 = template_alg.show()

        if inputs['model'] == 'BayeSampling':

            with col2:
                if len(file) == 2:
                    sp.vsfeatures = df_vs
                    st.info('You have upoaded the visula sample point file.')
                    feature_name = sp.features.columns.tolist()
                else:
                    feature_name = sp.features.columns.tolist()
                    mm = MinMaxScaler()
                    mm.fit(sp.features)
                    data_min = mm.data_min_
                    data_max = mm.data_max_
                    sp.trans_features = mm.transform(sp.features)
                    min_ratio, max_ratio = st.slider('sample space ratio', 0.8, 1.2, (1.0, 1.0))
        
                    sample_num = st.selectbox('sample number', ['10','20','50','80','100'])
                    feature_num = sp.trans_features.shape[1]

                    vs = np.linspace(min_ratio * data_min, max_ratio *data_max, int(sample_num))  

                    sp.vsfeatures = pd.DataFrame(vs, columns=feature_name)

                Bgolearn = BGOS.Bgolearn()

                Mymodel = Bgolearn.fit(data_matrix = sp.features, Measured_response = sp.targets, virtual_samples = sp.vsfeatures,
                                       opt_num=inputs['opt num'], min_search=inputs['min search'], noise_std= float(inputs['noise std']))
                # Mymodel = Bgolearn.fit(data_matrix = sp.features, Measured_response = sp.targets, virtual_samples = sp.vsfeatures)
                if inputs['sample criterion'] == 'Expected Improvement algorith':
                    res = Mymodel.EI()
                    
                if inputs['sample criterion'] == 'Expected improvement with "plugin"':
                    res = Mymodel.EI_plugin()

                if inputs['sample criterion'] == 'Augmented Expected Improvement':
                    with st.expander('EI HyperParamters'):
                        alpha = st.slider('alpha', 0.0, 3.0, 1.0)
                        tao = st.slider('tao',0.0, 1.0, 0.0)
                    res = Mymodel.Augmented_EI(alpha = alpha, tao = tao)

                if inputs['sample criterion'] == 'Expected Quantile Improvement':
                    with st.expander('EQI HyperParamters'):
                        beta= st.slider('beta',0.2, 0.8, 0.5)
                        tao = st.slider('tao_new',0.0, 1.0, 0.0)            
                    res = Mymodel.EQI(beta = beta,tao_new = tao)

                if inputs['sample criterion'] == 'Reinterpolation Expected Improvement':  
                    res = Mymodel.Reinterpolation_EI() 

                if inputs['sample criterion'] == 'Upper confidence bound':
                    with st.expander('UCB HyperParamters'):
                        alpha = st.slider('alpha', 0.0, 3.0, 1.0)
                    res = Mymodel.UCB(alpha=alpha)

                if inputs['sample criterion'] == 'Probability of Improvement':
                    with st.expander('PoI HyperParamters'):
                        tao = st.slider('tao',0.0, 0.3, 0.0)
                    res = Mymodel.PoI(tao = tao)

                if inputs['sample criterion'] == 'Predictive Entropy Search':
                    with st.expander('PES HyperParamters'):
                        sam_num = st.number_input('sample number',100, 1000, 500)
                    res = Mymodel.PES(sam_num = sam_num)  
                    
                if inputs['sample criterion'] == 'Knowledge Gradient':
                    with st.expander('Knowldge_G Hyperparameters'):
                        MC_num = st.number_input('MC number', 50,300,50)
                    res = Mymodel.Knowledge_G(MC_num = MC_num) 

                if inputs['sample criterion'] == 'Least Confidence':
                    
                    Mymodel = Bgolearn.fit(Mission='Classification', Classifier=inputs['Classifier'], data_matrix = sp.features, Measured_response = sp.targets, virtual_samples = sp.vsfeatures,
                                       opt_num=inputs['opt num'])
                    res = Mymodel.Least_cfd() 
       
                if inputs['sample criterion'] == 'Margin Sampling':
                    Mymodel = Bgolearn.fit(Mission='Classification', Classifier=inputs['Classifier'], data_matrix = sp.features, Measured_response = sp.targets, virtual_samples = sp.vsfeatures,
                            opt_num=inputs['opt num'])
                    res = Mymodel.Margin_S()

                if inputs['sample criterion'] == 'Entropy-based approach':
                    Mymodel = Bgolearn.fit(Mission='Classification', Classifier=inputs['Classifier'], data_matrix = sp.features, Measured_response = sp.targets, virtual_samples = sp.vsfeatures,
                            opt_num=inputs['opt num'])
                    res = Mymodel.Entropy()
            st.info('Recommmended Sample')
            sp.sample_point = pd.DataFrame(res[1], columns=feature_name)
            st.write(sp.sample_point)
            tmp_download_link = download_button(sp.sample_point, f'recommended samples.csv', button_text='download')
            st.markdown(tmp_download_link, unsafe_allow_html=True)


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


