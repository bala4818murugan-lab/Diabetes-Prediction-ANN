# diabetes_prediction_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import joblib
from keras.models import Sequential
from keras.layers import Dense


# Load the data
df=pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Documents\\kaggle data\\pima_diabetes_dataset\\diabetes_prediction_dataset.csv")

# ENCODER
le=LabelEncoder()
df['gender']=le.fit_transform(df['gender'])
df['smoking_history']=le.fit_transform(df['smoking_history'])

# input and output
X=df.drop(['diabetes'], axis=1)
y=df['diabetes']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# ann model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train,y_train, epochs=50, batch_size=32, validation_split=0.2)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

model.save("diabetes_prediction_model.keras")
