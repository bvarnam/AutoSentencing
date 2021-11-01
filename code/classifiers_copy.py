# imports

import pandas as pd

# modeling imports
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier, ExtraTreesClassifier,GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

# metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score

# As we start learning more classifying models, we'll keep adding those models
# to this list od dictionaries.
models = [{'knn':KNeighborsClassifier()},
        {'logreg':LogisticRegression()},
        {'dt':DecisionTreeClassifier()},
        {'bag':BaggingClassifier(random_state=42)},
        {'bag_knn':BaggingClassifier(random_state=42, n_estimators=10, base_estimator=KNeighborsClassifier())},
        {'bag_log':BaggingClassifier(random_state=42, n_estimators=10, base_estimator=LogisticRegression())},
        {'rf':RandomForestClassifier()},
        {'et':ExtraTreesClassifier()},
        {'ada':AdaBoostClassifier()},
        {'gboost':GradientBoostingClassifier()},
        {'svc':SVC()}
        ]

def classify(X_train, X_test, y_train, y_test):
    # Record all scores
    scores = []

    # train model
    for i in range(len(models)):
        for k,v in models[i].items():

            # fit/train models one ata time
            k = models[i][k]
            k.fit(X_train,y_train)

            # Predictions
            train_preds = k.predict(X_train)
            test_preds = k.predict(X_test)

            # Evaluate models
            train_acc = k.score(X_train,y_train)
            test_acc = k.score(X_test,y_test)
            acc_diff = abs(train_acc-test_acc)

            f1_train = f1_score(y_train,train_preds)
            f1_test = f1_score(y_test,test_preds)
            f1_diff = abs(f1_train-f1_test)

            train_pres = precision_score(y_train, train_preds)
            test_pres = precision_score(y_test, test_preds)
            pres_diff = abs(train_pres-test_pres)

            train_recall = recall_score(y_train,train_preds)
            test_recall = recall_score(y_test,test_preds)
            recall_diff = abs(train_recall-test_recall)

            # Appending scores
            scores.append([train_acc,test_acc,acc_diff,
                           f1_train,f1_test,f1_diff,
                           train_pres,test_pres,pres_diff,
                           train_recall, test_recall, recall_diff
                            ])





    scores_df = pd.DataFrame(scores,

    columns=['Train Acc','Test Acc','Acc-diff',
             'Train-F1','Test-F1','F1-diff',
             'Train-Pres','Test-Pres','Pres-diff',
             'Train_Recall','Test-Recall','Recall_diff'],

    index = ['knn','logreg','dt','bag',
    'bag_knn','bag_log','rf','et',
    'ada','gboost','svc'])

    return scores_df

# Add syntax that also outputs each models coefficents.
