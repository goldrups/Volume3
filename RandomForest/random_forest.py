"""
Random Forest Lab

Samuel Goldrup
Math 403
29 November 202
"""
import os
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
# import graphviz
# from uuid import uuid4

# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""

    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        if num_rows(sample) > 1:
            return sample[:,self.column] >= self.value #answer to question is true
        else:
            return sample[self.column] >= self.value

    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(self.value))

def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    l_mask = question.match(data)
    r_mask = ~l_mask
    left, right = data[l_mask], data[r_mask] #separate into two sets
    return left, right

def test_1():
    animals = np.loadtxt('animals.csv',delimiter=',')
    features = np.loadtxt('animal_features.csv',delimiter=',',dtype=str,comments=None)
    names = np.loadtxt('animal_names.csv',delimiter=',',dtype=str)
    question = Question(column=1,value=3,feature_names=features)
    left, right = partition(animals,question)
    assert len(left) == 62
    assert len(right) == 38

    question = Question(column=1, value=75, feature_names=features)
    left, right = partition(animals, question)
    assert len(left) == 0
    assert len(right) == 100

def test_2():
    animals = np.loadtxt('animals.csv', delimiter=',')
    assert gini(animals) == 0.4758
    assert info_gain(animals[:50], animals[50:], gini(animals)) == .14579999999999999


# Helper function
def num_rows(array):
    """ Returns the number of rows in a given array """
    if array is None:
        return 0
    elif len(array.shape) == 1:
        return 1
    else:
        return array.shape[0]

# Helper function
def class_counts(data):
    """ Returns a dictionary with the number of samples under each class label
        formatted {label : number_of_samples} """
    if len(data.shape) == 1: # If there's only one row
        return {data[-1] : 1}
    counts = {}
    for label in data[:,-1]:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


#Problem 2
def gini(data):
    """Return the Gini impurity of given array of data.
    Parameters:
        data (ndarray): data to examine
    Returns:
        (float): Gini impurity of the data"""
    counts = class_counts(data) #cardinality of each class
    N = num_rows(data)
    return 1 - np.sum([(count/N)**2 for count in counts.values()])

def info_gain(left, right, G):
    """Return the info gain of a partition of data.
    Parameters:
        left (ndarray): left split of data
        right (ndarray): right split of data
        G (float): Gini impurity of unsplit data
    Returns:
        (float): info gain of the data"""
    N = num_rows(left) + num_rows(right) #length of dataset
    return G - ((num_rows(left)/N)*gini(left)) - ((num_rows(right)/N)*gini(right))

# Problem 3, Problem 7
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 7
    Returns:
        (float): Best info gain
        (Question): Best question"""

    best_gain, best_question = 0, None
    features = feature_names[:-1]
    n = len(features)
    if random_subset:
        n_sqrt = int(np.floor(np.sqrt(n))) #optimal no. of features
        indices = np.random.randint(low=0,high=len(features),size=n_sqrt)

    G = gini(data)

    for i in range(n):
        if random_subset and i not in indices:
            continue
        unique_vals = list(set(data[:,i]))
        for val in unique_vals:
            question = Question(column=i,value=val,feature_names=feature_names)
            l,r = partition(data,question)
            if num_rows(l) < min_samples_leaf or num_rows(r) < min_samples_leaf:
                continue
            ig = info_gain(l,r,G)
            if ig > best_gain: #update the gain
                best_gain = ig
                best_question = question

    return best_gain,best_question

def test_3():
    animals = np.loadtxt('animals.csv',delimiter=',')
    features = np.loadtxt('animal_features.csv',delimiter=',',dtype=str,comments=None)
    bg,bq = find_best_split(animals,features)
    assert np.isclose(bg,0.12259833679833687)
    print(bq) #assert error with this...

# Problem 4
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data):
        self.prediction = class_counts(data) #just assign the attributes

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
        self.question = question #just assign the attributes
        self.left = left_branch
        self.right = right_branch


# Prolem 5
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    #enforce min_sample_leaf rule
    if num_rows(data) < 2*min_samples_leaf:
        return Leaf(data)
    
    opt_gain, corresp_q = find_best_split(data,feature_names,min_samples_leaf=min_samples_leaf,random_subset=random_subset)
    if opt_gain == 0 or current_depth >= max_depth:
        return Leaf(data)
    
    #use question to split
    left, right = partition(data,corresp_q)
    left_branch = build_tree(left,feature_names,min_samples_leaf,max_depth,current_depth=current_depth+1,random_subset=random_subset)
    right_branch = build_tree(right,feature_names,min_samples_leaf,max_depth,current_depth=current_depth+1,random_subset=random_subset)

    return Decision_Node(corresp_q,left_branch,right_branch)

# Problem 6
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    if isinstance(my_tree,Leaf): #if it is just a leaf, this is very simple
        return max(my_tree.prediction, key=my_tree.prediction.get)

    if my_tree.question.match(sample):
        return predict_tree(sample,my_tree.left)
    else:
        return predict_tree(sample,my_tree.right)

def analyze_tree(dataset,my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    N = num_rows(dataset)
    preds = []
    for datum in dataset: #see which predictions were right
        preds.append(predict_tree(datum,my_tree))

    preds = np.array(preds)
    correct_preds = sum(preds == dataset[:,-1])
    
    score = correct_preds / N
    return score

def test_4_5_6():
    animals = np.loadtxt('animals.csv',delimiter=',')
    animal_features = np.loadtxt('animal_features.csv',delimiter=',',dtype=str,comments=None)
    np.random.shuffle(animals)
    train, test = animals[:80,:],animals[80:,:]
    built_tree = build_tree(data=train,feature_names=animal_features)
    accuracy = analyze_tree(dataset=test,my_tree=built_tree)

    my_tree = build_tree(animals[:80],animal_features[:80])
    print(analyze_tree(animals[:20],my_tree))
    #assert analyze_tree(animals[:20],my_tree) == 0.9
    print(f'accuracy={accuracy}')


# Problem 7
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    #print(len(sample))
    all_preds = np.array([predict_tree(sample,tree) for tree in forest])
    voted_preds = np.mean(all_preds,axis=0) #count the votes
    preds = np.rint(voted_preds)
    
    return preds

def analyze_forest(dataset,forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    N = num_rows(dataset)
    preds = []
    for datum in dataset:
        preds.append(predict_forest(datum,forest))

    preds = np.array(preds)
    print(len(preds))
    correct_preds = sum(preds == dataset[:,-1])
    
    score = correct_preds / N
    return score

def test_7():
    animals = np.loadtxt('animals.csv',delimiter=',')
    animal_features = np.loadtxt('animal_features.csv',delimiter=',',dtype=str,comments=None)
    np.random.shuffle(animals)
    train, test = animals[:80,:],animals[80:,:]
    forest_gump = [build_tree(data=train,feature_names=animal_features,random_subset=True) for _ in range(10)]

    accuracy = analyze_forest(test,forest_gump)

    predict_forest(test,forest_gump)

    park = np.loadtxt("parkinsons.csv",delimiter=',')
    features = np.loadtxt("parkinsons_features.csv",delimiter=',',dtype=str)
    shuffled = park[:,1:]
    size = 100
    n = 5
    train = shuffled[:size]
    test = shuffled[size:size+30]
    forest = [build_tree(train,features,min_samples_leaf=15,random_subset=False) for _ in range(n)]
    print(predict_forest(test[0],forest))
    #assert predict_forest(test[0],forest) == 1.0

    print(f'accuracy={accuracy}')

# Problem 8
def prob8():
    """ Using the file parkinsons.csv, return three tuples. For tuples 1 and 2,
        randomly select 130 samples; use 100 for training and 30 for testing.
        For tuple 3, use the entire dataset with an 80-20 train-test split.
        Tuple 1:
            a) Your accuracy in a 5-tree forest with min_samples_leaf=15
                and max_depth=4
            b) The time it took to run your 5-tree forest
        Tuple 2:
            a) Scikit-Learn's accuracy in a 5-tree forest with
                min_samples_leaf=15 and max_depth=4
            b) The time it took to run that 5-tree forest
        Tuple 3:
            a) Scikit-Learn's accuracy in a forest with default parameters
            b) The time it took to run that forest with default parameters
    """
    parkinsons = np.loadtxt("parkinsons.csv",delimiter=',',dtype=float,comments=None)[:,1:]
    parkinsons_features = np.loadtxt('parkinsons_features.csv',delimiter=',',dtype=str,comments=None)[1:]
    np.random.shuffle(parkinsons)
    train,test = parkinsons[:100,:],parkinsons[100:130,:]

    #my own forest
    me_a = time.time()
    forest_gump = [build_tree(data=train,feature_names=parkinsons_features,min_samples_leaf=15,max_depth=4,random_subset=True) for _ in range(5)]
    me_time = time.time() - me_a
    me_acc = analyze_forest(dataset=test,forest=forest_gump)

    #sk learn forest on small set
    sk_a = time.time()
    sklearn_gump = RandomForestClassifier(n_estimators=5,max_depth=4,min_samples_leaf=15)
    sklearn_gump.fit(train[:,:-1],train[:,-1])
    sklearn_gump_acc = sklearn_gump.score(test[:,:-1],test[:,-1])
    sk_time = time.time() - sk_a

    #now do forest on the whole dataset
    N = num_rows(parkinsons)
    l = int(np.floor(0.8*N))
    train,test = parkinsons[:l,:],parkinsons[l:,:]
    sk_whole_time_a = time.time()
    sk_whole_forest = RandomForestClassifier()
    sk_whole_forest.fit(train[:,:-1],train[:,-1])
    sk_whole_time = time.time() - sk_whole_time_a
    sk_whole_acc = sk_whole_forest.score(test[:,:-1],test[:,-1])

    return (me_acc, me_time,sklearn_gump_acc,sk_time,sk_whole_acc,sk_whole_time)

def test_8():
    print(prob8())

## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if isinstance(my_tree, Leaf):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree):
    """Draws a tree"""
    #Remove the files if they already exist
    for file in ['Digraph.gv','Digraph.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf

def test_all():
    test_1()
    test_2()
    test_3()
    test_4_5_6()
    test_7()
    test_8()