from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

class MLModel:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f'Accuracy: {accuracy:.2f}')
        print('Classification Report:')
        print(report)
        print('Confusion Matrix:')
        print(conf_matrix)
        
        #filename = input('Model Filename: ')
        dump(self.model, 'trained_model.joblib')
        
        
    
    def preprocess(self, df, target_variable):
        label_encoder = preprocessing.LabelEncoder()
        scaler = preprocessing.StandardScaler()
   
    # Transform categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = label_encoder.fit_transform(df[col])

    # Scale numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        X = df.drop(target_variable, axis=1)
        y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)
    
        return X_train, X_test, y_train, y_test
