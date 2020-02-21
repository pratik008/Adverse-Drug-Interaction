import streamlit as st
import csv
from rdkit.Chem import Draw, MolFromSmiles
from scripts.inference import Inference
from scripts.helper import plot_horizonatal_bar_chart
import os

#Title
st.title("Deep Drug Interaction")

#Sidebar
if st.sidebar.checkbox('How does this work?'):
    st.sidebar.subheader('Short Description')
    st.sidebar.text("Data : 2 million Interactions \n from drugbank.com.")
    st.sidebar.text("Feature Engineering : Convert \n drug names to SMILES string.")
    st.sidebar.text("Deep Learning Model : CNN LSTM \n with Attention.")
    st.sidebar.text("Serving : streamlit app and \n command line.")

if st.sidebar.checkbox('Show Instructions'):
    st.sidebar.subheader('Instructions')
    st.sidebar.text("step 1 : Select drug A from candidate \n drug drop down.")
    st.sidebar.text("step 2 : Select drug B from target \n drug drop down.")
    st.sidebar.text("step 3 : Click on predict interactions")

if st.sidebar.checkbox('About the project'):
    st.sidebar.subheader('Project created by Pratik Mehta')
    st.sidebar.text('Github Link : \nhttp://bit.ly/DrugInteractionGithub')
    st.sidebar.text('Project Slides : \nhttp://bit.ly/DrugInteractionSlides')
    st.sidebar.text('Contact Email : pratik008@gmail.com')
    st.sidebar.text('Linkedin : linkedin.com/in/pratik008/')

#Main content
st.subheader("Welcome to Deep Drug Interaction")
st.write("Deep Drug Interaction is a deep learning tool designed to predict drug-drug interactions. "
         "Deep drug interactions is designed to help chemists to explore possible drug interactions of new molecules.")
st.write("More details, can be found in the online slides here : http://bit.ly/DrugInteractionSlides")

#Tutorail
st.subheader("How does this work?")
if st.checkbox('check this box to learn more.'):
    st.subheader('Tutorial')
    st.subheader('--------------------------------------------------------------------------------')
    st.subheader('Dataset - drugbank.com')
    st.write('2 million interaction pairs for 20,000 drug molecules is available from drugbank.com')
    st.image(os.path.join('helper_files','Drugbank.png'),width=600)
    st.subheader('--------------------------------------------------------------------------------')
    st.subheader('Molecule Structure')
    st.write('We can use the drug\'s molecular structure for making predictions. There are 47 different interaction types which include 99% of interactions.')
    st.image(os.path.join('helper_files', 'Structure.png'), width=600)
    st.subheader('--------------------------------------------------------------------------------')
    st.subheader('SMILES String')
    st.write('We can use the drug\'s SMILES representation as a proxy for molecular structure. We append the SMILES string for the 2 molecules - which acts as the feature space (X) and the corresponding interaction is the target (y).')
    st.image(os.path.join('helper_files', 'SMILES.png'), width=600)
    st.subheader('--------------------------------------------------------------------------------')
    st.subheader('ML Pipeline')
    st.write('Below is the Machine Learning Pipeline used for this project.')
    st.image(os.path.join('helper_files', 'ML_Pipeline.png'), width=600)
    st.subheader('--------------------------------------------------------------------------------')
    st.subheader('Architecture')
    st.write('A CNN LSTM with Attention architecture is used for this project.')
    st.image(os.path.join('helper_files', 'Architecture.png'), width=600)
    st.subheader('End of Tutorial')
    st.subheader('--------------------------------------------------------------------------------')

#Source code
st.subheader("Can I check out the source code?")
st.write("All source code is available at the drug interaction github repo: http://bit.ly/DrugInteractionGithub")

#Instructions
st.subheader("I am ready to run the prediction!")
if st.checkbox("Check this box to read the instructions."):
    st.subheader('Instructions')
    st.write("Here are the instructions to run the prediction.")
    st.write("Step 1 : Select drug A from candidate drug drop down.")
    st.write("Step 2 : Select drug B from target drug drop down.")
    st.write("Step 3 : Click on predict interactions.")

#Read drug list to load
drug_list = {}

smiles_dictionary_path = os.path.join('helper_files', 'smiles_dictionary.csv')
with open(smiles_dictionary_path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in readCSV:
        drug_list[row[0]] = row[1]
        i += 1
        if i > 1000:
            break

#Select Drug 1
st.subheader('Step 1 - Select Candidate Drug')
drug1 = st.selectbox(label='Select Candidate Drug', options=[''] + list(drug_list.keys()))
if drug1 != '':
    m = MolFromSmiles(drug_list[drug1])
    st.image(Draw.MolToImage(m))
    st.write(drug_list[drug1])

#Select Drug 2
st.subheader('Step 2 - Select Target Drug')
drug2 = st.selectbox(label='Select Target Drug', options=[''] + list(drug_list.keys()))
if drug2 != '':
    m = MolFromSmiles(drug_list[drug2])
    st.image(Draw.MolToImage(m))
    st.write(drug_list[drug2])

#Load and run Predictions
st.subheader('Step 3 - Predict Interactions')
if st.button("Predict Interactions"):
    '##'
    inference = Inference(druga_smiles=drug_list[drug1], drugb_smiles=drug_list[drug2], inference_model='mlp_train', inference_type='SMILES')
    top_labels, top_prob = inference.predict_interaction()
    top_prob = [l * 100 for l in top_prob]
    plot_horizonatal_bar_chart(top_prob, top_labels)





