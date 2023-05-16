"""
    Homework 1 Assignment from CS460G Spring 2023
    Author: Tyler Filbert
    Soruces: "https://medium.com/geekculture/
                step-by-step-decision-tree-id3-algorithm-from-
                scratch-in-python-no-fancy-library-4822bbfdd88f"
        This source I used as a guide to get started. 
        My implementation is different, but took inspiration on how
        to create and store a decision tree
"""


import os
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from math import log2
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import pprint


def read_pokemon_data(stats_path, is_lengendary_path, num_bins):
    """
    Returns a pandas dataframe consisting of stats_path and is_lengendary_path data
    combined into one dataframe. This dataframe is then binned into num_bins
    for features that can take on many different values

    stats_path and is_legendary_path are csv files
    num_bins is an integer
    """
    stats_df = pd.read_csv(stats_path)
    is_legendary_df = pd.read_csv(is_lengendary_path)
    stats_df['Legendary'] = is_legendary_df['Legendary']
    for col_name in stats_df:
        if not col_name == 'Legendary' and not col_name == 'Generation':

            # column values are not binary, discretize the column
            if len(stats_df[col_name].unique()) > 2:
                binned_vals = []
                min = stats_df[col_name].min()
                threshold_offset = (stats_df[col_name].max() - min) / num_bins
                for index, row in stats_df.iterrows():
                    for i in range(num_bins):
                        next_bin_threshold = (i+1) * threshold_offset + min
                        if stats_df[col_name].iloc[index] <= next_bin_threshold:
                            binned_vals.append(i)
                            break
                    else:
                        binned_vals.append(i)
                stats_df[col_name+'_Binned'] = pd.DataFrame(binned_vals)
                cat_type = CategoricalDtype(categories=range(num_bins), ordered=True)
                stats_df[col_name+'_Binned'] = stats_df[col_name+'_Binned'].astype(cat_type)

            # rename bianry columns anyways to make reading data easier    
            else: 
                stats_df[col_name+'_Binned'] = stats_df[col_name]

    return stats_df


def read_csv(csv_path, num_bins):
    """
    Returns a dataframe created from the data in csv_path. csv_path is data
    that has two Attribute columns and a class label column.
    Bin the attributes by num_bins

    csv_path is a csv file that contains two columns named Attribute1 and 2,
        and one that contains a Class Label column
    num_bins is an integer to discretize bins into
    """
    # Change to accept headers from pokemon data
    data = pd.read_csv(csv_path, names=['Attribute1', 'Attribute2', 'Class_Label'])

    #discretize
    for col_name in data:
        if col_name == 'Attribute1' or col_name == 'Attribute2':

            # column values are not binary, discretize the column
            if len(data[col_name].unique()) > 2:
                binned_vals = []
                min = data[col_name].min()
                threshold_offset = (data[col_name].max() - min) / num_bins
                for index, row in data.iterrows():
                    for i in range(num_bins):
                        next_bin_threshold = (i+1) * threshold_offset + min
                        if data[col_name].iloc[index] <= next_bin_threshold:
                            binned_vals.append(i)
                            break
                    else:
                        binned_vals.append(i)
                data[col_name+'_Binned'] = pd.DataFrame(binned_vals)
                cat_type = CategoricalDtype(categories=range(num_bins), ordered=True)
                data[col_name+'_Binned'] = data[col_name+'_Binned'].astype(cat_type)

    return data


def get_total_entropy(examples, class_label, class_results):
    """
    Calcualte the total entropy on the examples dataframe for the class_label column

    examples is a pandas dataframe with attributes and at least one class label
    class_label is column name to be used as the class label
    class_results is a list of values the class label can be
    """
    total_rows = len(examples[class_label])
    entropy = 0

    for result in class_results:
        total_option_count = len(examples[examples[class_label] == result])
        total_option_entr = -(total_option_count/total_rows)*log2(total_option_count/total_rows)
        entropy += total_option_entr

    return entropy


def get_feature_entropy(feature_val_data, class_label, class_results):
    """
    Calculate the entropy for a feature on feature_val_data for class_label

    feature_val_data is data that has a unique value that a feature can take on,
        inclduing other data in said rows
    class_label is the label in the dataframe for the class
    class_results is a list of possible values the class can be 
    """
    total_rows = len(feature_val_data)
    entropy = 0

    for result in class_results:
        class_result_count = len(feature_val_data[feature_val_data[class_label] == result])
        result_entropy = 0
        if class_result_count != 0:
            result_entropy = - (class_result_count/total_rows)*log2(class_result_count/total_rows)
        entropy += result_entropy

    return entropy


def get_information_gain(examples, feature, class_label, class_results):
    """
    Calculate the information gain for the feature in the examples dataframe
    on the class_label class

    examples is a dataframe with attributes and at least one class label
    feature is feature name to find information gain of
    class_label is the label in the dataframe for the class
    class_results is a list of possible values the class can be
    """
    unique_feature_vals = examples[feature].unique()
    total_rows = len(examples)
    info_gain = 0.0

    for feature_val in unique_feature_vals:
        feature_val_data = examples[examples[feature] == feature_val]
        feature_val_count = len(feature_val_data)
        feature_val_entropy = get_feature_entropy(feature_val_data, class_label, class_results)
        info_gain += feature_val_count/total_rows * feature_val_entropy
    
    feature_info_gain = get_total_entropy(examples, class_label, class_results) - info_gain
    return feature_info_gain


def get_max_info_feature(examples, class_label, class_results):
    """
    Iterate through all features in examples. Return the name of the feature
    that has the highest information gain

    examples is a dataframe with attributes and at least one class label
    class_label is the label in the dataframe for the class
    class_results is a list of possible values the class can be
    """
    features = [col for col in examples.columns if 'Binned' in col]

    max_info_gain = -1
    max_info_feature = None

    for feature in features:
        feature_info_gain = get_information_gain(examples, feature, class_label, class_results)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature

    return max_info_feature


def make_sub_tree(examples, feature, class_label, class_results):
    """
    Creates a subtree (dictionary) for feature from examples dataframe
    Tries to find pure datasets, if they are not pure flag the edge as
    NP (not pure)
    Returns a dictionary that has feature values and what their result is
        (not pure or pure and the value)

    examples is a dataframe with attributes and at least one class label
    feature is the feature name to split the data on
    class_label is the label in the dataframe for the class
    class_results is a list of possible values the class can be
    """
    # Count values that feature takes on and how many times
    feature_val_counts = examples[feature].value_counts(sort=False)
    tree = {}
    
    for feature_value, count in feature_val_counts.iteritems():
        # cut the data to only include entries that have the current feature value
        feature_value_data = examples[examples[feature] == feature_value]
        is_node = False

        for result in class_results:
            #how many entries for this current feature value have each class result
            class_count = len(feature_value_data[feature_value_data[class_label] == result])
            
            # if all entries for this feature value have the same class result
            # it is pure, save it to the tree
            if class_count == count:
                tree[feature_value] = result 
                # update data to not include data that was just added to the tree
                examples = examples[examples[feature] != feature_value] 
                is_node = True

        # the current value for the feature is not pure
        # cannot decide anything, flag that this deicion is not pure
        if not is_node:
            tree[feature_value] = 'NP'
        
    return tree, examples


def get_most_common_class_result(examples, class_label, class_results):
    """
    Given a dataframe with attributes and a class_label, find the most
    common value that the class takes on.
    Return the most common result

    examples is a pandas dataframe with attributes and at least one class label
    class_label is column name to be used as the class label
    class_results is a list of values the class label can be
    """
    most_common_class_result = None
    max = 0
    for result in class_results:
        current_result_count = len(examples[examples[class_label] == result])
        if current_result_count > max:
            most_common_class_result = result
            max = current_result_count
    return most_common_class_result


def make_tree(root, prev_feat_val, examples, class_label, class_results, depth, max_depth):
    """
    Recursively create the most informative decision tree possible given a max_depth

    root is a dictionary that contains what feature values go where (i.e prediction or another question)
    prev_feat_val is the name of the feature that was previously used to split data (parent)
    examples is a dataframe containing all data that has yet to be added to the tree
    class_label is column name to be used as the class label
    class_results is a list of values the class label can be
    depth is the current depth of the tree
    max_depth is an integer that specifies how deep the tree can be at maximum
    """
    if len(examples) != 0: # there is data remaining to split
        max_info_feat = get_max_info_feature(examples, class_label, class_results)
        max_info_feat_val = get_information_gain(examples, max_info_feat, class_label, class_results)
        tree, examples = make_sub_tree(examples, max_info_feat, class_label, class_results)
        next_root = None

        # There is no more info to gain, decide by choosing most common class label left
        # and stop operations on this branch
        if max_info_feat_val == 0:
            for node, branch in tree.items():
                if branch == 'NP':
                    root[prev_feat_val] = get_most_common_class_result(
                                                examples[examples[max_info_feat] == node], 
                                                class_label, 
                                                class_results) 
            return
        
        # There was a parent feature
        elif prev_feat_val != None:
            # if the max depth has been reached, guess on the remianing branches and save to see
            # then return to stop operations on this branch
            if depth == max_depth:
                finish_tree = {}
                for node, branch in tree.copy().items():
                    if branch == 'NP':
                        finish_tree[node] = get_most_common_class_result(
                                                            examples[examples[max_info_feat] == node], 
                                                            class_label, 
                                                            class_results)
                    else:
                        finish_tree[node] = branch
                root[prev_feat_val] = {max_info_feat: finish_tree}
                return

            # add the subtree to the tree. attach to parent (prev_feat_val)
            else:
                root[prev_feat_val] = {}
                root[prev_feat_val][max_info_feat] = tree
                next_root = root[prev_feat_val][max_info_feat]

        # There was not a parent feature, add current feature as root
        else:
            root[max_info_feat] = tree
            next_root = root[max_info_feat]
        
        # dispurse into the children that were created, making their own subtrees if
        # the branch is marked as not pure
        for node, branch in next_root.items():
            if branch == 'NP':
                feat_val_data = examples[examples[max_info_feat] == node]
                make_tree(next_root, node, feat_val_data, class_label, class_results, depth+1, max_depth)


def id3(examples, class_label):
    """
    Declare the data to be passed to make a decision tree
    Starts running the id3 algorithm

    examples is a dataframe to create a decision tree off of
    class_label is the class name to create predictions for in the tree
    """
    data = examples.copy()
    tree = {}
    class_results = data[class_label].unique()
    make_tree(tree, None, data, class_label, class_results, 0, 3)
    return tree


def predict_value(dec_tree, datapoint):
    """
    Given a decision tree (dec_tree), iterate through the datapoint's feature
    values to predict the class result

    dec_tree is a completed decision tree in the form of a dictionary
    datapoint is the feature values of an instance and its class result
    """
    # the point is a leaf, return it
    if not isinstance(dec_tree, dict): 
        return dec_tree
    
    # haven't found a leaf, traverse through tree
    root = next(iter(dec_tree))
    feature_val = datapoint[root]
    if feature_val in dec_tree[root]:
        return predict_value(dec_tree[root][feature_val], datapoint)
    
    return None


def test_tree(test_data, dec_tree, class_label):
    """
    Iterate through predicting all instances in test_data on dec_tree for
    class class_label and return the total accuracy

    test_data is a pandas dataframe that holds feature and class data
    dec_tree is a completed decision tree in the form of a dictionary
    class_label is the class name to create predictions for in the tree
    """
    total_correct = 0
    total = 0
    for index, row in test_data.iterrows():
        result = predict_value(dec_tree, test_data.iloc[index])
        if result == test_data[class_label].iloc[index]:
            total_correct += 1
        total += 1

    return total_correct / total


def get_error(examples, dec_tree, class_label):
    """
    Get the training data set error. Iterate through training set, and predicting class values
    based on training set attribute values.

    examples is the training data as a pandas dictionary
    dec_tree is a finished decision tree as a dictionary
    class_label is the class label to check prediction from
    """
    diff = 0
    total = len(examples)
    for index, row in examples.iterrows():
        prediction = predict_value(dec_tree, examples.iloc[index])
        diff += ((1 if examples[class_label].iloc[index] else 0) - (1 if prediction else 0))**2
    return diff/total


def get_graph_values(dec_tree, num_bins):
    """
    Predict each possible combination of bin values for attribute 1 and 2
    Save these to a dictionary and return that dictionary

    dec_tree is a deicison tree modeled in a dictionary
    num_bins is an integer that declares the number of bins for both attributes
    """
    graph = {}
    for i in range(num_bins):
        graph[i] = {}
        for j in range(num_bins):
            df = pd.DataFrame(columns=[
                                     'Attribute1',
                                     'Attribute2',
                                     'Class_Label',
                                     'Attribute1_Binned',
                                     'Attribute2_Binned'])
            df.loc[0] = np.array([i, j, None, i, j], dtype=np.float64)
            graph[i][j] = predict_value(dec_tree, df.iloc[0])
    return graph


def plot_graph(graph_values, dataset_name):
    """
    Display a graph that has each pair of attribute values as a scatter plot
    On the same plot, shade in each square on the grid with the values of the
    prediction to use as a decision surface

    graph_values is a dictionary with first key as x value, second key as y value, value as
    whether or not pokemon is lengendary (1 or 0)
    """
    #Covert passed dictionary to 2d numpy array
    data = np.zeros(shape=(len(graph_values),len(graph_values)))
    for xcord in graph_values:
        row_vals = []
        for ycord in graph_values[xcord]:
            row_vals.append(True if graph_values[ycord][xcord] else False)
        data[xcord] = row_vals
    data = np.ma.masked_where(data == False, data)

    #Create plot
    cmap = colors.ListedColormap(['green'])
    cmap.set_bad(color='red')
    fig, ax = plt.subplots()
    ax.pcolormesh(data, cmap=cmap, edgecolor='black', linestyle='-', lw=1)

    #Scatter plot each point. Assign 1 to green, 0 to red
    for xcord in graph_values:
        for ycord in graph_values:
            if graph_values[xcord][ycord] == 1:
                plt.plot(
                        xcord, 
                        ycord, 
                        'o', 
                        color='green',
                        markeredgecolor='black')
            else:
                plt.plot(
                        xcord, 
                        ycord, 
                        'o', 
                        color='red',
                        markeredgecolor='black')
                
    #Add labels to graph
    ax.set_xticks(np.arange(0,len(graph_values), 1))
    ax.set_yticks(np.arange(0,len(graph_values), 1))
    plt.title('Decision surface of {data} decision tree.'.format(data=dataset_name))
    plt.xlabel('Attribute 1 Value')
    plt.ylabel('Attribute 2 Value')
    red_patch = mpatches.Patch(color='red', label='0')
    green_patch = mpatches.Patch(color='green', label='1')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()


def main():
    """DECLARE HOW MANY BINS TO DISCRETIZE DATA HERE. 10 WORKS WELL AND IS THE DEFAULT"""
    num_bins = 10

    """ *UNCOMMENT BELOW TO RUN DECISION TREE ON POKEMON
        *ENTER YOUR PATH TO THE POKEMON STATS AND POKEMON LEGENDARY FILES
            (IN THAT ORDER) TO THE READ_POKEMON_DATA FUNCTION
        *OPTIONALLY UNCOMMENT THE WITH OPEN BLOCK (PROVIDE A FILE PATH TO THE OPEN CALL)
            TO SEE A FORMATTED PRINT OUT OF THE FINAL DECISION TREE"""
    # pokemon_data = read_pokemon_data(
    #                     os.path.join(os.path.dirname(__file__), 'data/pokemonStats.csv'),
    #                     os.path.join(os.path.dirname(__file__), 'data/pokemonLegendary.csv'),
    #                     num_bins)
    
    # dec_tree = id3(pokemon_data, 'Legendary')
    # # with open(os.path.join(os.path.dirname(__file__), 'output.txt'),'w') as f:
    # #     pp = pprint.PrettyPrinter(indent=1, width=80, stream=f)
    # #     pp.pprint(dec_tree)

    # print('accuracy {acc}%'.format(acc=test_tree(pokemon_data, dec_tree, 'Legendary')*100))
    # print('error {err}%'.format(err=get_error(pokemon_data, dec_tree, 'Legendary')*100))


    """ *UNCOMMENT BELOW TO RUN DEICIOSN TREE AND PRINT GRAPH ON SYNTHETIC DATA
        *ENTER YOUR PATH TO THE SYNTHETIC DATA INTO THE READ_CSV FUNCTION
        *ENTER THE SYNTHETIC DATASET IN THE PLOT_GRAPH CALL TO GET THE CORRECT DATASET NAME
            IN GRAPH TITLE
        *OPTIONALLY UNCOMMENT THE WITH OPEN BLOCK (PROVIDE A FILE PATH TO THE OPEN CALL)
            TO SEE A FORMATTED PRINT OUT OF THE FINAL DECISION TREE"""
    # examples = read_csv(os.path.join(os.path.dirname(__file__), 'data/synthetic-1.csv'), num_bins)
    # dec_tree = id3(examples, 'Class_Label')

    # # with open(os.path.join(os.path.dirname(__file__), 'output.txt'),'w') as f:
    # #     pp = pprint.PrettyPrinter(indent=1, width=80, stream=f)
    # #     pp.pprint(dec_tree)

    # print('accuracy {acc}%'.format(acc=test_tree(examples, dec_tree, 'Class_Label')*100))
    # print('error {err}%'.format(err=get_error(examples, dec_tree, 'Class_Label')*100))
    # plot_graph(get_graph_values(dec_tree, num_bins), 'synthetic1')

if __name__ == '__main__':
    main()