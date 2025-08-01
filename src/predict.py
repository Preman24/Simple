import pandas as pd
from model import MLModel

df = pd.read_csv('..\\data\\Iris.csv')
df = df.iloc[:,1:]
df['Species'] = df['Species'].astype('category')


# Initialize the model
model = MLModel()

# Preprocess the data
X_train, X_test, y_train, y_test = model.preprocess(df, target_variable='Species')
print('1 pass')
# Train the model
model.train(X_train, y_train)
print('2 passs')

# Evaluate the model
model.evaluate(X_test, y_test)
print('3 pass')
