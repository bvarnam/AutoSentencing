# Executive Summary
## Methodology
As stated above, our group created a variety of classification models, ranging from logistic regression to neural networks, using data that included demographic information and data that excluded such information. We trained these models on data taken provided to the public by the United States Sentencing Commission and compared the relative performances (specifically the accuracy scores and misclassification rates) of these sets of models to determine whether demographic information had predictive power regarding sentencing.
## Results
While all of our models performed slightly better with demographic information than without, the difference in performance was never substantial enough to warrant us definitively saying that demographic information held the type of predictive power we initially thought it might.
## Conclusions and Recommendations
The drop in accuracy across all models when removing demographic information is not enough relative to the number of features to decisively prove there is major discrimination in sentencing aspect of the judicial systems. This drop does show that there is some effect in including demographic features, however it is likely because of the amount of demographic features relative to our total number of features. When reviewing which cases were misclassified by our model as No-Prison when in actuality the offender received a Prison sentence (False Positives), we found no substantial difference in models containing demographic variables versus those without. 
We recommend that our Non-Profit partners use this as an initial filter for possible cases, and apply their own domain knowledge when qualitatively investigating the false negative misclassifications for lawsuits and sentence reductions based on the grounds of discrimination.
## Webapp
This app was designed for use by anyone representing a person convicted of a criminal offense. The app will then predict whether the defendant will receive prison time vs a non-prison time. This predictive tool can be used to assess the likelihood that a case was misclassified or mishandled, thereby helping lawyers focus on appealing those cases that fall into these categories.

The app works by using the following pretrained classification models. Below are the Accuracy scores for each model.

K Nearest Neighbors:
>Train Acc: 0.98
 
> Test Acc: 0.97

Random Forest:
>Train Acc: 1

>Test Acc: 0.99

Extra Trees:
>Train Acc: 1

>Test Acc: 0.99

Bagging Classification:
>Train Acc: 0.99

>Test Acc: 0.99

Decision Trees:
>Train Acc: 1

>Test Acc: 0.99

All of our models **(judges)** were trained on data from the United States Sentencing Commission taken in the fiscal year 2020. Each of these classifiers behave like judges who each have a vote, 0 for non-prison sentence or 1 for prison sentence. Each vote is a prediction by each model, and at the end of the script the votes are tallied up. If the sum of the tallies is greater or equal to 3, the collective prediciton is a prison sentence, anything else would be considered a non-prison sentence.