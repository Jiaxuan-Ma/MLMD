import streamlit as st

# Define possible modelsd in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "GradientBoostingClassifier",
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
    
    st.info('TO SOLVE **CLASSIFICATION**')
    
    # st.write('training')

    # st.write("No additional parameters")
    col1, col2 = st.columns([2,2])
    with col1:
        with st.expander("Hyper Parameter"):
            inputs['learning rate'] = st.number_input('learning rate',0.001,10.0,0.1)
            inputs['nestimators'] = st.number_input('number estimators',1, 1000, 100)
            inputs['max features'] = st.selectbox('max features', ['auto', 'sqrt', 'log2'])
            max_depth = st.checkbox('max depth', None)
            inputs['max depth'] = None
            if max_depth:
                inputs['max depth'] = st.number_input('max depth',1, 100, 3)
            random_state = st.checkbox('random state 1024',True)
            if random_state:
                inputs['random state'] = 1024
            else:
                inputs['random state'] = None

    return inputs,col2


# To test the alg independent of the app or template, just run 
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()
    
    