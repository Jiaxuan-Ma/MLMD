import streamlit as st

# Define possible modelsd in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "LogisticRegression",
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
    
    st.info('TO TUNE **HYPERPARAMETERs**')
    
    # st.write('training')
    tab1, tab2 = st.tabs(['BASIC','GRID SEARCH'])
    with tab1:
    # st.write("No additional parameters")
        col1, col2 = st.columns([2,2])
        with col1:
            with st.expander("Hyper Parameter"):
                
                inputs['penalty'] = st.selectbox('penalty',('l2','l1','elasticnet','None'))
            
                # inputs['C'] = st.number_input('C', 0, 100000, 1)

                # inputs['max iter'] = st.number_input('max iter', 0, 100000, 100)

                random_state = st.checkbox('random state 1024',True)
                if random_state:
                    inputs['random state'] = 1024
                else:
                    inputs['random state'] = None
                options_parameter = st.radio('Tune Hyper parameters', ['C','max iter'])
                inputs['tune_parameter'] = options_parameter
    return inputs,col2,tab2


# To test the alg independent of the app or template, just run 
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()
    
    