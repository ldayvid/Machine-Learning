import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn import tree

data = pd.read_csv('food.csv')

Input = data.drop(columns=['favourite food'])
Output = data['favourite food']

# Train with 75% of data and test with 25%
Intrain, Intest, Outtrain, Outtest = train_test_split(Input, Output, test_size=0.25)

# Create decision tree
model = DecisionTreeClassifier()
model.fit(Intrain, Outtrain)

joblib.dump(model, 'food-recommender.joblib')
model = joblib.load('food-recommender.joblib')

predictions = model.predict(Intest)

score = accuracy_score(Outtest, predictions)

# Export decision tree as graph
tree.export_graphviz(model, out_file='food.dot', 
                    feature_names=['age', 'gender'], 
                    class_names=sorted(Output.unique()),
                    label='all',
                    rounded=True,
                    filled=True)
