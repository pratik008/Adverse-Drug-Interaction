import boto3
import botocore
import lxml.etree as et
import xml.etree.ElementTree as ET


# Contains functions to read data from XML file obtained from DrugBank.
# Separate functions to read locally and to read from S3 storage.
#
# Defines a Drug class that has name, structure
# and interactions as attributes.
# 
# Also defines an Interaction class that contains the names of
# two drugs and a textual description
# of their interaction.

class Drug:
    '''Contains relevant features of each drug

    Attributes :
        name (str): Name of drug
        structure (str): SMILES representation of drug
        interactions (list): List of 
    '''
    def __init__(self, name, structure, interactions):
        self.name = name
        self.structure = structure
        self.interactions = interactions

    def __repr__(self):
        return 'Name : {}\nStructure : {}'.format(self.name, self.structure)
    
    def has_structure(self):
        if self.structure is not None:
            return True
        else:
            return False


class Interaction:
    '''Contains a drug pair and a textual description of the interaction.

    Attributes :
        druga (str): Name of first drug
        drugb (str): Name of second drug
        description (str): Text about the effect of taking the drug together
    '''

    def __init__(self, druga = None, drugb = None, description = None):
        self.druga = druga
        self.drugb = drugb
        self.description = description
    
    def assign(self, druga, drugb, description):
        self.druga = druga
        self.drugb = drugb
        self.description = description


def read_from_bucket(bucket = 'insight-ashwin-s3-bucket', \
    key = 'drugbank_data/drugbank_data.xml', number_of_drugs = 50000):
    '''Function to read an XML file from AWS S3 storage

    Args : 
        bucket (str): Name of bucket on S3
        key (str): Location of XML file in the bucket
        number_of_drugs (int): Total number of drugs to be read

    Returns :
        drug_list (list): List of Drug instances initialized with data read.
    '''

    s3 = boto3.resource('s3', use_ssl = False, verify = False)
    obj = s3.Object(bucket, key)
    file = obj.get()['Body'].read()
    tree = et.ElementTree(et.fromstring(file))
    return read_data(tree, number_of_drugs)


def read_from_file(xml_file, number_of_drugs = 50000):
    '''Function to read an XML file locally

    Args :
        xml_file (str): Name of XML file
        number_of_drugs (int): Total number of drugs to be read

    Returns :
        drug_list (list): List of Drug instances initialized with data read.
    '''
    tree = ET.parse(xml_file)
    return read_data(tree, number_of_drugs)


def read_data(tree, number_of_drugs = 50000, addon = '{http://www.drugbank.ca}'):
    '''Function to read an XML file given the tree object.

    Args :
        tree (ElementTree): ElementTree instance of the XML file to be read
        number_of_drugs (int): Total number of drugs to be read
        addon (str): default text to be added to XML tags

    Returns :
        drug_list (list): List of Drug instances initialized with data read.
        smiles_dict (dict): Dictionary that maps drug names to their SMILES strings.
    '''
    root = tree.getroot()
    drug_count = 0
    drug_list = []
    smiles_dict = {}
    for elem in root:
        structure = ''
        interactions = {}

        # go through the drug descriptions and extract relevant features of each drug : 
        for i in range(len(elem)):
            if elem[i].tag == addon+'name':
                name = (elem[i].text).lower()
                
            if elem[i].tag == addon+'calculated-properties':
                calculated_properties = elem[i]
                for calc_property in calculated_properties:
                    if calc_property[0].text == 'SMILES':
                        structure = calc_property[1].text
                        
            if elem[i].tag ==  addon+'drug-interactions':
                interactions_list = elem[i]
                interactions = {inter[1].text.lower():inter[2].text.lower()\
                     for inter in interactions_list}

        if structure != '' and interactions != {}:
            drug = Drug(name, structure, interactions)
            drug_list.append(drug)
            smiles_dict[name] = structure
            drug_count += 1
            if drug_count >= number_of_drugs:
                break
    
    return drug_list, smiles_dict


def generate_interactions(drug_list, smiles_dict):
    '''Function to extract interaction pairs from a list of drugs

    Args :
        drug_list (list): List of Drug instances initialized with data read.
        smiles_dict (dict): Dictionary that maps drug names to their SMILES strings.

    Returns :
        interaction_list (list): List of Interaction instances extracted from drug_list.
    '''
    interaction_list = []
    interaction_count = 0
    for drug in drug_list:
        for drugb, description in drug.interactions.items():
            # Only keep drugs that have structural data
            if drugb in smiles_dict:
                interaction = Interaction(drug.name, drugb, description)
                interaction_list.append(interaction)
                interaction_count += 1

    return interaction_list