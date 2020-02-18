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
    st.sidebar.subheader('Short tutorial')
    st.sidebar.text('Here is a short tutorial')
if st.sidebar.checkbox('Show Instructions'):
    st.sidebar.subheader('The instructions')
    st.sidebar.text('Here are the instructions')
if st.sidebar.checkbox('About the project'):
    st.sidebar.subheader('Project created by Pratik Mehta')
    st.sidebar.text('Contact Details')

#Main content
st.subheader("Welcome to Deep Drug Interaction")
st.write("Deep Drug Interaction is a deep learning tool designed to predict drug-drug interactions. "
         "Deep drug interactions is designed to help chemists to explore possible drug interactions of new molecules.")
st.write("More details, can be found in the online slides here : Link")

#Tutorail
st.subheader("How does this work?")
if st.checkbox('check this box to learn more.'):
    st.subheader('Tutorial')
    st.text('Here is a quick guide to understand the inner workings.')

#Source code
st.subheader("Can I check out the source code?")
st.write("All source code is available at the drug interaction github repo: Link")

#Instructions
st.subheader("I am ready to run the prediction!")
if st.checkbox("Check this box to read the instructions."):
    st.subheader('Instructions')
    st.write("Here are the instructions to run the prediction.")

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





