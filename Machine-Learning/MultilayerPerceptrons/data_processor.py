"""
Read in CS460G Programming Assignment2 Data
"""
import numpy as np
import csv
import os

class DataProcessor:
    def __init__(self):
        self.all_sets = {}

    
    def get_example_values(self, csv_file):
        """
        Given a csv file save the contents to a dictioanry
        The dictioanry will contain the index of the image (0, 1, etc.)
        as the key, and the value will contain a dictionary
        that holds the class label and the image values
        """
        try:
            with open(csv_file, newline='') as f:
                reader = csv.reader(f)
                
                for row in reader:
                    example_vals = {}
                    class_label = row[0]
                    image_vals = []
                    for val in row[1:]:
                        image_vals.append(int(val)/255)
                    example_vals['class_label'] = class_label
                    example_vals['image_vals'] = image_vals
                    self.all_sets[len(self.all_sets)] = example_vals
            print('Completed reading in file!\n')
                    
        except OSError:
            print('Invalid csv file')
            exit()
