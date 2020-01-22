from labels import *
import csv

# File to extract particular keywords from textual description of interactions.
#


class Relation:
    '''Contains information about the cause and effect relationship between two drugs.

    Attributes :
        subject (str): Name of subject drug.
        object (str): Name of object drug.
        description (str): Textual description of interaction between subject and object.
        relation (str): Text that contains only the interaction text extracted from 
            description.
        normalized_relation (str): Keyword related to relation.
    '''

    def __init__(self, interaction = None):
        if interaction is None:
            self.subject = None
            self.object = None
            self.description = None
            self.relation = None
            self.normalized_relation = None
        else:
            self.get_relation_from_interaction(interaction)
            self.get_normalized_relation(NORMALIZED_KEYWORDS)
    
    def get_relation_from_interaction(self, interaction):
        '''Get relation information from Interaction instance.

        Args:
            interaction (class Interaction): Interaction instance to read drug pair and
                interaction description from.
        '''

        druga = interaction.druga
        drugb = interaction.drugb
        self.description = interaction.description.lower()

        index1 = self.description.find(druga)
        index2 = self.description.find(drugb)

        prefix = ''
        # Find which drug appears first in the interaction description.
        if min(index1, index2) != 0:
            prefix = self.description[:min(index1, index2)]

        if index1 < index2:
            self.subject = druga
            self.object = drugb
            self.relation = prefix + ' ' + self.description[index1 + len(druga): index2].strip()
        else:
            self.subject = drugb
            self.object = druga
            self.relation = prefix + ' ' + self.description[index2 + len(drugb): index1].strip()
        
        if prefix != '':
            self.subject, self.object = self.object, self.subject

    def is_in_order(self, keywords):
        '''Check if keywords appear in the correct order in relation.
        
        '''

        if len(keywords) == 0:
            return True

        index = [self.relation.find(k) for k in keywords]
        
        if index[0] == -1:
            return False
        for i in range(1, len(index)):
            if index[i] == -1 or index[i] < index[i-1]:
                return False
        return True

    def get_normalized_relation(self, normalized_keywords):
        for keywords in normalized_keywords:
            if self.is_in_order(keywords.split()):
                self.normalized_relation = keywords
                return
        
        self.normalized_relation = None

    def is_normalized(self):
        if self.normalized_relation is not None:
            return True
        else:
            return False

    def get(self):
        '''Return the subject, object and normalized_relation of the instance.'''

        return self.subject, self.object, self.normalized_relation


def generate_relations(interaction_list):
    '''Extract keyword and relational data from interaction description.
    
    '''
    relation_list = []
    for interaction in interaction_list:
        relation = Relation(interaction)
        relation_list.append(relation)
    
    return relation_list
    

def remove_duplicates(relation_list):
    '''Remove possible duplicate relations.

    Check all drug pairs and filter them out if the same pair (being agnostic 
    about the order) appear with the same interaction. Retain copies if they 
    have different interactions.

    Args :
        relation_list (list): List of Relation instances to be filtered.

    Returns :
        new_relation_list (list): Filtered list of Relation instances.
        filter_count (int): Number of elements filtered.
    '''

    relation_dict = {}
    filter_count = 0
    new_relation_list = []
    for relation in relation_list:
        relation_pair = frozenset([relation.subject, relation.object])
        if relation_pair not in relation_dict:
            relation_dict[relation_pair] = relation.normalized_relation
            new_relation_list.append(relation)
        elif relation_dict[relation_pair] != relation.normalized_relation:
            new_relation_list.append(relation)
        else:
            filter_count += 1
    
    return new_relation_list, filter_count


def filter_unknowns(relation_list):
    '''Filter out Relation objects that don't have a normalized relation

    Args :
        relation_list (list): List of Relation instances to be filtered.

    Returns :
        new_relation_list (list): Filtered list of Relation instances.
        filter_count (int): Number of elements filtered.
    '''

    filter_count = 0
    new_relation_list = []
    for relation in relation_list:
        if relation.normalized_relation is not None:
            new_relation_list.append(relation)
        elif relation.normalized_relation is None:
            filter_count += 1
        else:
            print(relation.normalized_relation)
    return new_relation_list, filter_count
    # TODO : Output interactions (relation.relation) not being picked up


def generate_labels(relation_list, save = False):
    '''Generate numerical labels for each interaction


    '''

    label_map = {}
    label_lookup = {}
    counter = 0
    for relation in relation_list:
        if relation.normalized_relation not in label_map:
            label_map[relation.normalized_relation] = counter
            label_lookup[counter] = relation.normalized_relation
            counter += 1

    if save:
        write_dict_to_csv(label_lookup, 'label_lookup.csv')
    
    return label_map, label_lookup

def write_dict_to_csv(d, csv_file):
    '''Write a dictionary to a csv file

    Args :
        d (dict): Dictionary to write into file.
        csv_file (str): Name and location of csv file to be written to
    
    Returns :
        None
    '''
    
    if csv_file[-3:] == 'csv':
        try:
            with open(csv_file, 'w') as file:
                w = csv.writer(file)
                w.writerows(d.items())
        except IOError:
            print('I/O Error')
    else:
        print('Not a csv file.')