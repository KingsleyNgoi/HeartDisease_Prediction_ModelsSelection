# HeartDisease_Prediction_ModelsSelection
## 1. | Introduction 👋
  * Dataset Problems 🤔 </br>
    * 👉 This dataset contains information about contains diagnoses of heart disease patients. </br>
    * 👉 The <mark><b>goal of the problem is to predict the target variable</b></mark>, called condition.
    * 👉 It has <mark><b>only two unique associated values</b></mark>, so let's treat this a  <mark><b>binary classification problem</b></mark>.
    * 👉 Choose the best machine learning model with highest accuracy <mark><b>to determine whether a person has heart disease or not</b></mark>.
  * Machine Learning Modules 👨‍💻 </br>
  👉 The <b>models</b> used in this notebook:
    <ol start="1">
        <li> <b>Logistic Regression</b>,</li>
        <li> <b>K-Nearest Neighbour (KNN)</b>,</li>
        <li> <b>Support Vector Machine (SVM)</b>,</li>
        <li> <b>Gaussian Naive Bayes</b>,</li>
        <li> <b>Decision Tree</b>,</li>
        <li> <b>Random Forest</b>,</li>
        <li> <b>Gradient Boosting</b>,</li>
        <li> <b>AdaBoost</b>, and</li>
        <li> <b><span style="font-size: 8; background-color: #7289da;"><sup>*NEW*</sup></span> XGBoosting</b>.</li>
    </ol>
  * Dataset Description 🧾 </br>
  👉 There are <mark><b>14 variables</b></mark> in this dataset:
    <ul>
        <li> <b>9 categorical</b> variables, and</li>
        <li> <b>5 continuous</b> variables.</li>
    </ul>

    * <mark><b>FEATURES</b></mark>
    <ol start="1">
      <li> <b>age |</b> age of persons</b></li>
      <li> <b>sex |</b> Gender of patient (Male:0/Female:1)</b></li>
      <li> <b>cp | </b> Chest Pain type (4 values)</b></li>
    <ul>
                        <li> Value 0: typical angina</br></li>
                        <li> Value 1: atypical angina</br></li>
                        <li> Value 2: non-anginal pain</br></li>
                        <li> Value 3: asymptomatic</br></li>
    </ul>
      <li> <b>trtbps |</b> resting blood pressure (in mm Hg)</b></li>
      <li> <b>chol |</b> serum cholestrol in mg/dl fetched via BMI sensor</b></li>
      <li> <b>fbs |</b> fasting blood sugar > 120 mg/dl (1 = true; 0 = false)</b></li>
      <li> <b>restecg |</b> Resting Electrocardiographic (ECG) results (values 0,1,2)</b></li>
    <ul>
                        <li> Value 0: normal</br></li>
                        <li> Value 1: having ST-T wave abnormality</br>
                             (T wave inversions and/or ST elevation or depression of > 0.05 mV)</br></li>
                        <li> Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria</br></li>
    </ul>
      <li> <b>thalachh |</b> Maximum Heart Rate Achieved</b></li>
      <li> <b>exng |</b> exercise induced angina (1 = yes; 0 = no)</b></li>
      <li> <b>oldpeak |</b> oldpeak = ST depression induced by exercise relative to rest <br>
      <li> <b>slp |</b> the slope of the peak exercise ST segment</b></li>
      <li> <b>caa |</b> number of major vessels (0-3) colored by flourosopy</b></li>
      <li> <b>thall |</b> Thalium stress test results: 0=normal, 1=fixed defect, 2 = reversable defect</b></li> 
    </ol>

    * <mark><b>TARGET VARIABLE</b></mark></br>
        * condition: diagnosis of heart disease (angiographic disease status)<br>
          > <b>Value 0:</b> < 50% diameter narrowing (negative for disease) <br>
          > <b>Value 1:</b> > 50% diameter narrowing (positive for disease)
</div><br>

## 2. | File Descriptions 👓
- `heart.csv`: the dataset file.
- `HeartDisease_Prediction_KingsleyNgoi.ipynb`: contains the code of data exploration, preparation and modeling. 
- `heart_disease_ada_best_model.pkl`: the classification model. 

## 3. | Accuracy of Best Model 🧪
AdaBoost Classifier
- Train Accuracy achieved: 87.30%
- Test Accuracy achieved: 91.23%

## 4. | Conclusion 📤
- In this study respectively,
- We have tried to a predict classification problem in Heart Disease Dataset by a variety of models to classifiy Heart Disease predictions in the contex of determining whether anybody is likely to get hearth disease based on the input parameters like gender, age and various test results or not.
- We have made the detailed exploratory analysis (EDA).
- There have been NO missing values in the Dataset and removing 1 duplicate value.
- We have decided which metrics will be used.
- We have analyzed both target and features in detail.
- We have perform SMOTE oversampling to minority target class to treat the imbalance dataset condition.
- We have transformed categorical variables into dummies so we can use them in the models.
- We have handled with skewness problem for make them closer to normal distribution with examing the distribution with skewness value, kurtosis value, the boxplot, histogram and qqplot.
- We have used pipeline, stratifiedKFold and cross-checked the models obtained from train sets by applying cross validation for each model performance and hyperparameter tuning and best paramters selection.
- We have examined the feature importance of some models.
- Lastly we have examined the results of all models visually with respect to select the best one which is AdaBoost with dropping high extraordinarily chi2 score features for the problem in hand.

## 5. | Reference 🔗
<ul><b><u>Kaggle Notebook 📚</u></b>
        <li><a style="color: #3D5A80" href="https://www.kaggle.com/code/caesarmario/listen-to-your-heart-a-disease-prediction">Listen to Your Heart: A Disease Prediction by MARIO CAESAR</a></li>
        <li><a style="color: #3D5A80" href="https://www.kaggle.com/code/azizozmen/heart-failure-predict-8-classification-techniques/notebook">Heart_Failure_Predict_8_Classification_Techniques by MATTHEW CONNOR</a></li>
</ul>
<ul><b><u>Online Articles 🌏</u></b>
        <li><a style="color: #3D5A80" href="https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5">5 SMOTE Techniques for Oversampling your Imbalance Data by Cornellius Yudha Wijaya</a></li>
        <li><a style="color: #3D5A80" href="https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/">Tune Hyperparameters for Classification Machine Learning Algorithms by  Jason Brownlee </a></li>
</ul>
<ul><b><u>Models Parameters </u></b>
        <li><a style="color: #3D5A80" href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">sklearn.linear_model.LogisticRegression by scikit-learn developers (BSD License).</a></li>
        <li><a style="color: #3D5A80" href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">sklearn.neighbors.KNeighborsClassifier by scikit-learn developers (BSD License).</a></li>
        <li><a style="color: #3D5A80" href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">sklearn.svm.SVC by scikit-learn developers (BSD License).</a></li>
        <li><a style="color: #3D5A80" href="https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html">sklearn.naive_bayes.GaussianNB by scikit-learn developers (BSD License).</a></li>
        <li><a style="color: #3D5A80" href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">sklearn.tree.DecisionTreeClassifier by scikit-learn developers (BSD License).</a></li>
        <li><a style="color: #3D5A80" href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">sklearn.neighbors.KNeighborsClassifier by scikit-learn developers (BSD License).</a></li>
        <li><a style="color: #3D5A80" href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">sklearn.ensemble.RandomForestClassifier by scikit-learn developers (BSD License).</a></li>
        <li><a style="color: #3D5A80" href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html">sklearn.ensemble.GradientBoostingClassifier by scikit-learn developers (BSD License).</a></li>
        <li><a style="color: #3D5A80" href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html">sklearn.ensemble.AdaBoostClassifier by scikit-learn developers (BSD License).</a></li>
        <li><a style="color: #3D5A80" href="https://xgboost.readthedocs.io/en/stable/parameter.html">XGBoost Parameters by xgboost developers..</a></li>
</ul>
<ul><b><u>Chi2 Statistical Testing for Feature Selection </u></b>
        <li><a style="color: #3D5A80" href="https://machinelearningmastery.com/feature-selection-with-categorical-data/">How to Perform Feature Selection with Categorical Data by Jason Brownlee.</a></li>
        <li><a style="color: #3D5A80" href="https://towardsdatascience.com/using-the-chi-squared-test-for-feature-selection-with-implementation-b15a4dad93f1">Using the Chi-Squared test for feature selection with implementations by Dr.Saptarsi Goswami. </a></li>
</ul>
<ul><b><u>Notebook Styling </u></b>
        <li><a style="color: #3D5A80" href="https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html">Table Visualization: Styler Object and Customising the Display by NumFOCUS, Inc. Hosted by OVHcloud.</a></li>
</ul>


