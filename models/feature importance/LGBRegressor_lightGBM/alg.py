import streamlit as st

# Define possible modelsd in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "LGBRegressor",
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
    
    st.info('TO SOLVE **FEATURE IMPOTRANCE RANK**')
    
    # st.write('training')
    col1, col2 = st.columns([2,2])
    with col1:
        # st.write("No additional parameters")
        with st.expander("Hyper Patameter"):
            inputs['num_leaves'] = st.number_input('leaves number',36) 
            inputs['metric'] = st.selectbox('metric',['l1', 'l2','auc'])
            inputs['learning rate'] = st.selectbox('learning rate',options = [1e-2,1e-1,1e-3,1e-4])
            inputs['feature rate'] = st.slider('feature fraction',0.0, 1.0, 0.4)
            inputs['bagging rate'] = st.slider('bagging fraction',0.0, 1.0, 0.7)
            random_state = st.checkbox('random state 1024',True)
            if random_state:
                inputs['random state'] = 1024
            else:
                inputs['random state'] = None
            inputs['lambda'] = st.slider('lambda',0.0, 1.0, 0.7)
            inputs['max depth'] = st.number_input('max depth',10)
            inputs['is unbalance'] = st.checkbox('unbalance',True)
    
            inputs['niterations'] = st.number_input('number iteraiton',5)
            inputs['nestimators'] = st.number_input('number estimators',1000)
        early_stopping = st.checkbox('early stopping', True)

            # inputs['verbose'] = st.select_slider('verbose',[2,-1,0,1])
        with st.expander("early_stopping"):
            if not early_stopping:
                inputs['early stopping'] = False
            else:
                inputs['early stopping'] = True
                inputs['early stopping rounds'] = st.number_input('early stopping rounds',100)
                inputs['test size'] = st.slider('test size',0.0, 1.0, 0.2)
            

    # st.write('visualizations')
    # inputs["visualization_tool"] = st.selectbox("How to log metrics?", ("Not at all", "Tensorboard", "comet.ml"))

    # if inputs["visualization_tool"] == "comet.ml":
    #     # TODO: Add a tracker how many people click on this link.
    #     "[Sign up for comet.ml](https://www.comet.ml/) :comet: "
    #     inputs["comet_api_key"] = st.text_input("Comet API key (required)")
    #     inputs["comet_project"] = st.text_input("Comet project name (optional)")
    # elif inputs["visualization_tool"] == "Tensorboard":
    #     st.markdown(
    #         "<sup>Logs are saved to timestamped dir in `./logs`. View by running: `tensorboard --logdir=./logs`</sup>",
    #         unsafe_allow_html=True,)
    
    return inputs, col2


# To test the alg independent of the app or template, just run 
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()
    
    