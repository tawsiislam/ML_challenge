from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, StackingClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

def stacking_eval(X_train,X_test,Y_train,Y_test):
    """ This function is used for evaluating the model and finding the best possible Stacking classifier.
    Return the accuracy score and can return the best parameters achieved from RandomSearchCV
    """
    # random_grid used to test almost all possible parameters for fine tuning
    random_grid = {'rf__max_depth': [40,50,60,80],
                        'rf__min_samples_leaf': [1,2,4],
                        'rf__min_samples_split': [2,5,10],
                        'rf__n_estimators': [200,250,300],
                        'mlp__hidden_layer_sizes': [(14,14,14),(16,16,16),(18,18,18),(21,21,21)],
                        'mlp__max_iter': [250,300,400]}
    # All parameters found through RandomizedSearchCV
    classifiers = [('qda',QuadraticDiscriminantAnalysis()),
                   ('rf',RandomForestClassifier(criterion='entropy',n_estimators=250,min_samples_split=10,min_samples_leaf=1,
                                                max_depth=40,max_features='sqrt')),
                   ('mlp',MLPClassifier(hidden_layer_sizes=(18,18,18),alpha=0.1, activation='logistic', solver='adam',max_iter=250))]

    stack_model = StackingClassifier(classifiers,final_estimator=LinearDiscriminantAnalysis(),cv=10)
    #random_model = RandomizedSearchCV(estimator=stack_model, param_distributions=random_grid, n_iter = 150, cv = 3, verbose = 4, n_jobs = -1)
    stack_model.fit(X_train,Y_train)

    Y_pred = stack_model.predict(X_test)
    #best_params = random_model.best_params_    # Used to get the best parameters from RandomSearchCV
    score = accuracy_score(Y_test,Y_pred)
    return score

def stacking_get_labels(X_train,X_test,Y_train):
    """ This function is used for predicting the labels in the Evaluation data set with the Stacking 
    classifier and parameters found above. Returns the labels of the evaluation data
    """
    classifiers = [('qda',QuadraticDiscriminantAnalysis()),
                   ('rf',RandomForestClassifier(criterion='entropy',n_estimators=250,min_samples_split=10,min_samples_leaf=1,
                                                max_depth=40,max_features='sqrt')),
                   ('mlp',MLPClassifier(hidden_layer_sizes=(18,18,18),alpha=0.1, activation='logistic', solver='adam',max_iter=250))]
    stack_model = StackingClassifier(classifiers,final_estimator=LinearDiscriminantAnalysis(),cv=10)
    stack_model.fit(X_train,Y_train)
    Y_pred = stack_model.predict(X_test)
    return Y_pred