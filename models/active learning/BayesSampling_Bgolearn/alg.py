import streamlit as st

# Define possible modelsd in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "BayeSampling",
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
    
    st.info('TO SOLVE **SAMPLING**')
    
    # st.write('training')

    # st.write("No additional parameters")
    col1, col2 = st.columns([2,2])
    with col1:
        with st.expander("Hyper Parameter"):
            inputs['mission'] = st.selectbox('mission',['Regression'])
            # if inputs['mission'] == 'Regression':
            inputs['Regressor'] = st.selectbox('regressor',['GaussianProcess'])
            inputs['noise std'] = st.selectbox('noise std',['0.01','0.001','0.0001','heteroheneous'])
                            
            inputs['opt num'] = st.number_input('sample number',0, 10, 1)
            inputs['min search'] = st.selectbox('min search',[False, True])

            inputs['sample criterion'] = st.selectbox('sample criterion',['Expected Improvement algorith',
                    'Expected improvement with "plugin"','Augmented Expected Improvement','Expected Quantile Improvement',
                    'Reinterpolation Expected Improvement','Upper confidence bound','Probability of Improvement',
                    'Predictive Entropy Search','Knowledge Gradient'])

            # if inputs['mission'] == 'Classification':
            #     inputs['Classifier'] = st.selectbox('classifier',['GaussianProcess','LogisticRegression','NativeBayes','SVM','RandomForest'])
            #     inputs['noise std'] = st.selectbox('noise std',['0.001','0.0001','0.00001','0.000001'])
            #     inputs['min search'] = False
            #     inputs['opt num'] = st.number_input('sample number',0, 10, 1)
            #     inputs['sample criterion'] = st.selectbox('sample criterion',['Least Confidence', 'Margin Sampling', 'Entropy-based approach'])



    return inputs,col2


# To test the alg independent of the app or template, just run 
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()
    
    