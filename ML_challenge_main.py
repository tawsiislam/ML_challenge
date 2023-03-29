import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from stacking import stacking_eval, stacking_get_labels

# Reading the CSV files
train_data_csv = "TrainOnMe.csv"
df = pd.read_csv(train_data_csv)
scaler = StandardScaler()

eval_data_csv = "EvaluateOnMe.csv"
eval_df = pd.read_csv(eval_data_csv)

# Correcting misspelled labels, values and 
df["y"].replace(["ragspel","yckelharpa","erpent"],["Dragspel","Nyckelharpa","Serpent"],inplace=True)
df["x11"].replace(["Tru","F"],["True","False"],inplace=True)
df["x12"].replace(["Flase","F"],"False",inplace=True)
df["x6"].replace(["Ostra stationen",np.nan],["Östra stationen","Unknown"],inplace=True)
df['x5'] = df['x5'].astype(str).replace(['0.0','-0.0'],[0,1]).astype(float) # Rewriting -0 and 0
df = df.loc[(df['x5']== 0) | (df['x5'] == 1)]   # For x5

# Translating True and False, and 0 and -0 to ints
lbe = LabelEncoder()
df['x11'] = lbe.fit_transform(df['x11'])
eval_df['x11'] = lbe.fit_transform(eval_df['x11'])
df['x12'] = lbe.fit_transform(df['x12'])
df['x5'] = lbe.fit_transform(df['x5'])
#df['x6'] = lbe.fit_transform(df['x6'])  # Not used, Could be done with LabelEncoder or OneHotEncoder

labels = df["y"].reset_index(drop=True) # Collecting labels
df = df.drop(columns='y')   # Removing the labels from the training data

#Scaling on numbers-only columns
scaler = StandardScaler()
df[["x1","x2","x3","x4","x7","x8","x9","x10",'x13']] = scaler.fit_transform(df[["x1","x2","x3","x4","x7","x8","x9","x10",'x13']])
eval_df[['x1','x2','x3','x4','x7','x8','x9','x10','x13']] = scaler.transform(eval_df[['x1','x2','x3','x4','x7','x8','x9','x10','x13']])
training_data = df[['x1','x2','x3', 'x4','x7','x8', 'x9', 'x10','x11','x13']]
eval_data = eval_df[['x1','x2','x3', 'x4','x7','x8', 'x9', 'x10','x11','x13']]


def evaluate_model(selected_data,labels):
    X_train,X_test,y_train,y_test = train_test_split(selected_data,labels,train_size=0.8,stratify=labels)   # Assuming evaluation data has same proportion on the labels as training
    # kf = KFold(n_splits=5,shuffle=True,random_state=50)
    # split = kf.split(selected_data,labels)
    score_list = []

    # For-loop used for K-fold cross-validation
    # for train_ind, test_ind in split:
    #     X_train, X_test = selected_data.iloc[train_ind], selected_data.iloc[test_ind]
    #     y_train, y_test = labels[train_ind], labels[test_ind]
    #     score_list.append(stacking(X_train,X_test,y_train,y_test))
    score_list = stacking_eval(X_train,X_test,y_train,y_test)    # Trains the model, can also be used for RandomSearchCV to get best_params
    #print(best)
    #print("All score: ",score_list)
    print("Mean: ",np.mean(score_list))
    #print("Std: ",np.std(score_list))

#le = LabelEncoder()
#evaluate_model(training_data,labels)   # Used for evaluating model

y_pred = stacking_get_labels(training_data,eval_data,labels)    # Used for classifying evaluation data

y_pred_df = pd.DataFrame(data=y_pred,columns=None)
y_pred_df.to_csv('final_labels.txt', sep = ',', index = False,header=False)
