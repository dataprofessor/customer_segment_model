## importing required libraries
from pycaret.clustering import *
import streamlit as st
import pandas as pd
from PIL import Image

## loading the kmeans model
model = load_model('Final Kmeans Model')

## defining a function to make predictions
def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    predictions = predictions_df['Cluster'][0]
    return predictions

## defining the main function
def run():

    ## loading an image
    image = Image.open('customer_segmentation.png')

    ## adding the image to the webapp
    st.image(image,use_column_width=True)

    ## adding a selectbox making a choice between two broadways to predict new data points
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    ## adding some information about the app's functioning to the sidebar
    st.sidebar.info('This app is created to segment customer based on their behavior')

    ## adding the title for the streamlit app
    st.title("Customer Segmentation Prediction App")

    ## adding steps to be followed if the user selects Online mode of prediction 
    if add_selectbox == 'Online':

        ## adding a number input box to get age value
        Age = st.number_input('Age', min_value=18, max_value=100, value=25)

        ## adding a number input box to get income value
        Income = st.number_input('Income', min_value=9000, max_value=200000, value=20000)

        ## adding a number input box to get spending score value
        SpendingScore = st.number_input('SpendingScore', min_value=0.0 , max_value=1.0, format="%.2f")

        ## adding a number input box to get savings value
        Savings = st.number_input('Savings', min_value=0.0 , max_value = 25000.0, format="%.2f")

        ## defining the output variable 
        output=""

        ## creating a input dictionary with all the input features
        input_dict = {'Age' : Age, 'Income' : Income, 'SpendingScore' : SpendingScore, 'Savings' : Savings}

        ## converting the input dictionary into a pandas dataframe
        input_df = pd.DataFrame([input_dict])

        ## adding a button to make predictions when clicked on by the user
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output =  str(output)

        ## displaying the output after successful prediction
        st.success('The output is {}'.format(output))

    ## adding steps to be followed if the user selects Batch mode of prediction
    if add_selectbox == 'Batch':

        ## adding a file uploader button for the user to upload the csv file containing data points
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        ## block of code to be run once a csv file is uplloaded by the user
        if file_upload is not None:

            ## reading the csv file using pandas
            data = pd.read_csv(file_upload)

            ## making predictions
            predictions = predict_model(model,data=data)

            ## writing the predicitons
            st.write(predictions)

## calling the main function
if __name__ == '__main__':
    run()