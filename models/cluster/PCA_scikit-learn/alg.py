import streamlit as st

# Define possible modelsd in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "PCA",
}

# LightGBM can use -- categorical features -- as input directly. It doesn’t need to convert 
# to one-hot encoding, and is much faster than one-hot encoding (about 8x speed-up).

def show():
    """Shows the components for the template and returns user inputs as dict."""
    
    # `show()` is the only method required in this module. You can add any other code 
    # you like above or below. 
    
    inputs = {}  # dict to store all user inputs until return
    
    # with st.sidebar:
        
    # Render all template-specific sidebar components here. 

    # Use ## to denote sections. Common sections for training templates: 
    # Model, Input data, Preprocessing, Training, Visualizations
    # Store all user inputs in the `inputs` dict. This will be passed to the code
    # template later.
    inputs["model"] = MODEL["model"]
    
    # st.write("preprocessing")
    # inputs["Normalize"] = st.selectbox('normalize method', ['Z-Score Standardization','Min-Max Scale'])
    
    st.info('TO SOLVE **DIM REDUCTION**')
    
    # st.write('training')
    col1, col2 = st.columns([2,2])
    with col1:
        # st.write("No additional parameters")
        with st.expander("Hyper Patameter"):
            
            inputs['ncomponents'] = st.number_input('number components', 2)
            inputs['normalize'] = st.selectbox('normalize', ['MinMaxScaler'])

    return inputs, col2


# To test the alg independent of the app or template, just run 
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()
    
    
    





    # elif inputs['model'] == 'PCA':
    #     from sklearn.decomposition import PCA
    #     from sklearn.preprocessing import StandardScaler
    #     fs.normalize_features = StandardScaler().fit_transform(fs.features)
    #     pca = PCA(n_components=inputs['ncomponents'])
    #     with col2:
    #         if st.button('train'):
    #             principalComponents = pca.fit_transform(fs.normalize_features)
    #             # principalDF = pd.DataFrame(data = principalComponents)
    #             principalDf  = pd.DataFrame(principalComponents)
    #             new_columns = ['PCA_{}'.format(i+1) for i in range(len(principalDf.columns))]
    #             principalDf.columns = new_columns
    #             st.write(principalDf)
    #             explained_varianceDF = pd.DataFrame(pca.explained_variance_ratio_)
    #             explained_varianceDF.index = new_columns
    #             explained_varianceDF.columns = ['explained_variance']
    #             st.write(explained_varianceDF)