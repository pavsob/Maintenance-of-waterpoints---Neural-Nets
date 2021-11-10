Running the code instructions

To run particular task navigate to the src folder in the command window and write python main.py <task_id> <train|test|predict> <other_arguments> 
where task ID, action type and other are arguments.

There are four types of task parameter: task1, task2, task3 and task4. Each task has three action argument train, test and predict.
Folders already contain a trained model, but if the user changes any parameters, the model must be trained before testing or predicting.

Task 4 is about solving imbalanced data set problems; therefore, there is a third argument that specifies what method to use to process imbalanced data. 
These arguments are over (for over-sampling), under (for under-sampling), and smote (for SMOTE over-sampling technique).

Example of a command for calling task4: python main.py task4 train smote

For task 4, there always have to be the third argument; otherwise, it will throw an error. If the smote model is trained, then test and predict also must have smote argument.
As default there is saved smote model. If the user wants to use different technique, it must be re-trained.

**************************
List of all imported libraries:
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import sys
import numpy as np
### imblearm
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE