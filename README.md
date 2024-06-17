# Predictive Analysis of In-App Purchase Availability and Transparency
## **Introduction**
***Overview of Dataset***
>- The project trained ensemble models using 2M data entries from the "2017~2020 Google Play Store Applications" dataset by majority
voting method and compared metrics with individual models to aid investors in decision-making.
>- I used the "Google Play Store Applications" dataset collected in June 2021 in our final project. The initial dataset had 2,312,944 entries and 24 features.

***Data Dictionary***

<img src="graphs/datadictionary.png" alt="Graph" width="500">

After reviewing the features of the initial dataset, I am curious about the availability and transparency of “In-App Purchases” and see the role it plays in shaping user preferences, app strategies, and the overall competitiveness and success of mobile applications in the market.
>***Use Cases:***
>- Role: Software Investor
>- Goal: Assists investors in identifying and leveraging the features of mobile apps that maximize their benefits.
>- Approach: Identify the best model for maximizing your company's benefits by predicting whether or not an app has "in-app purchases," allowing investors to develop a more effective strategy for finding the apps with the highest ROI.

>***Model Selection:***
>Given that our target variable, "In-App Purchase," and the majority of variables in the dataset are categorical values (True/False), we >conducted three tree-based classifier models in this report.
>- Random Forest Classifier
>- Decision Tree Classifier
>- Gradient Boosting Classifier

>***Metrics for Model Evaluation(Classification Model):***
>- Confusion Matrix
>- Precision / Recall / F-1 Score
>- Receiver Operating Characteristic Curve (ROC) and Area Under the Curve (AUC)
>- Precision-Recall Curve

Precision is a crucial metric in this report. In the mobile app store market, the cost of false positives (predicting In-App purchases when they won't occur) is significant. Therefore, our prediction strategy focuses on minimizing false positives and ensuring accurate positive predictions to maximize profits from In-App purchases.
In contrast, the cost of false negatives (missed In-App purchases) is not high in our case, as we assume the cost of encouraging purchases is relatively low. Even if a purchase would have occurred naturally without our efforts, the additional cost is not substantial.
## **Data Resampling and Cleaning**

***Undersampling***

I first checked the data distribution of the target variable “In-App Purchases” and the variable contained the class imbalance issue. The percentage of "false" is 91.6%, significantly higher than the 8.44% of "true." Therefore, we resample the data by using undersampling.
<img src="graphs/before_unsam.png" alt="Graph" width="500">

After the undersampling, we changed the proportion of “false” and “true” to 2:1. Since a specific percentage difference can accurately represent the real data distribution, we have chosen to retain some data imbalance and have not adjusted the ratio between the two categories to make them equal.

<img src="graphs/after_unsam.png" alt="Graph" width="500">

***Droping Columns***

I dropped the columns that are highly unique or contain non-related information, such as URL or email address.

<img src="graphs/drop_column.png" alt="Graph" width="500">

***Missing Values***

I examined the dataset for null values, and the table below displays the total count of missing values for each variable that contains null values.

<img src="graphs/missing_value.png" alt="Graph" width="500">

## **Data Transform and Imputation**

The dataset contained complicated categorical data, and many columns required preprocessing to be used for analysis and modeling.

***Extraction and Converting Data for Analysis***

<img src="graphs/extract_convert.png" alt="Graph" width="500">

***Data Encoding***

I applied **LabelEncoder()** from sklearn to encode the categorical values.
<img src="graphs/data_encode.png" alt="Graph" width="500">

***Outlier***

I checked the outlier of numerical variables using **IQR value and box plot**.
<img src="graphs/outlier.png" alt="Graph" width="500">

<img src="graphs/outlier_1.png" alt="Graph" width="500">

Additionally, I only processed the outliers in **Install_Avg** and retain outliers in other fields to accurately represent real-world app store installation situations.

## **Exploratory Data Analysis**
After initializing the data cleaning and imputation, we conducted an exploration of the variables.

>"Install_Avg" and "In-App Purchases" relationship: When "In-App Purchases" is 0, the average "Install Avg" is significantly higher at 130,870, compared to 1,195,096 when "In-App Purchases" is 1.
<img src="graphs/eda_1.png" alt="Graph" width="500">

>"Size_numeric(KB)" and "In-App Purchases" relationship: Apps with "In-App Purchases" set to 1 have a larger average size of 34,122 compared to 18,781 for apps with "In-App Purchases" set to 0.
<img src="graphs/eda_2.png" alt="Graph" width="500">

>"Rating Count" and purchase correlation: Apps with more rating counts are more likely to be purchased. When "In-App Purchases" is 0, the average "Rating Count" is 2.10, while it's 3.27 when "In-App Purchases" is 1.
<img src="graphs/eda_3.png" alt="Graph" width="500">

>Influence of "Min version": The average values are similar for different "In-App Purchases" values, but apps with "In-App Purchases" set to 1 tend to support slightly higher versions.
<img src="graphs/eda_4.png" alt="Graph" width="500">

>Impact of “Ad Supported”: Approximately 56 percent of apps have advertisements, and those with ads have a purchase ratio of 43 percent, significantly higher than the 20 percent purchase rate for ad-free apps.
<img src="graphs/eda_5.png" alt="Graph" width="500">

>"Editor's Choice" and purchase rate: Only 0.1 percent of apps are chosen by editors. For editor's choice apps, the purchase rate is 94 percent, while apps not chosen by editors have a 33 percent purchase rate.
<img src="graphs/eda_6.png" alt="Graph" width="500">

>- In the correlation heatmap, we find that " Update interval days", "Ratings" and "Rating Count" are strongly and positively correlated, with correlation coefficients close to 1.
>- "In-App Purchases" is positively correlated with "Rating", "Rating Count", "Ad Support", "Update interval days" and "size_numeric(KB)" with a correlation around 0.25.
>- "Price" and "Free" are negatively correlated with a correlation coefficient around -0.25.
<img src="graphs/eda_7.png" alt="Graph" width="500">

In addition, we observed that when apps had a higher number of rating counts, they were more likely to be purchased.
## **Data Modeling Preprocessing**
***Target and Feature variables***
<img src="graphs/target_feature.png" alt="Graph" width="500">

***Data Splitting and Normalization***

We imported the train_test_split function from sklearn.model_selection to randomly split the train and test datasets. Import StandardScaler from sklearn.preprocessing to normalize feature variables.
<img src="graphs/data_split_normal.png" alt="Graph" width="500">

## **Random Forest Classifier**

***Hyperparameter Tunning (Cross Validation)***
In hyperparameter tuning, we import the **RandomizedSearchCV** function from **sklearn.model_selection** to optimize the model's performance. We apply the best hyperparameter on the train datasets (X_train, y_train).
<img src="graphs/rf_1.png" alt="Graph" width="500">

Pairs of parameters were examined using cross-validation, and the best parameters were selected based on the 'Precision' score, which considers balanced model performance.

***Model_Tuning (Controlling imBalance)***
The target variable exhibited an imbalance, with the number of data points labeled as '1' constituting 33% of the total data, while '0' accounted for the rest. While the imbalance was not severe, we sought to enhance the model's performance by applying class weights and **SMOTE (Synthetic Minority Over-sampling Technique)**.

After applying class weights and SMOTE, the F1 score improved; however, our primary metric of interest, 'precision,' did not exhibit the same improvement. As a result, we concluded that the basic Random Forest model, without any further tuning, performed best for our purposes.
<img src="graphs/rf_2.png" alt="Graph" width="500">

***Model Evaluation***
The best random forest model has a precision of 0.73, a recall of 0.53, and an F1-score of 0.61. Comparatively, higher precision and lower recall indicate a large number of False Negatives. As seen in the confusion matrix, there are 13,688 positives ('purchased') misclassified as negatives ('no purchased'). In comparison, the number of False Positives is low at 5512. This model exhibits higher precision and comparably lower recall.
<img src="graphs/rf_3.png" alt="Graph" width="500">

Additionally, we examined the ROC curve, which enables us to assess the trade-offs between True Positive Rates and False Positive Rates. The ROC curve allows us to compare the TPR and FPR. The TPR was 0.52, and the FPR was 0.473. Both scores were similar, but the curve displayed a noticeable deviation from the top-left corner (0,1), indicating a gap from the ideal score of 1. When considering the slightly higher TPR, the model is more effective at correctly identifying positive cases, thereby minimizing false negatives.
<img src="graphs/rf_4.png" alt="Graph" width="500">

The Precision-Recall curve (PR curve) reflects poor performance in terms of recall. The recall score is low at 0.53, resulting in a significant gap between the top-right corner (1,1) and the curve. Ideally, the curve should closely approach the corner, but this gap indicates a substantial distance between the corner and the curve. This reflects the model's poor recall scores, indicating that it struggles to properly distinguish positive values.
Additionally, it took **1.53 seconds to train this model**.

***Feature Importance***
In the Random forest, the following features were considered important features in the specified order: ‘Update Interval Days’, 'Rating', 'Min Version', ‘Category’, Rating Count', ‘Editors Choice’, 'Free', 'size_numeric(KB)', 'Ad supported', ‘Currency’, ‘Price','Installs_average' and , 'Content Rating'.

Update Interval Days' plays a significant role in reducing impurity, suggesting that it's a strong predictor for splitting the data at the root node or top-level nodes of the decision trees in the forest. 'Rating' is the second most important feature for reducing impurity, indicating its importance in making decisions in the trees. On the other hand, 'Price,' 'Installs_average,' and 'Content Rating' reduce less impurity, while the other features show similar impacts on reducing impurity.

The 'Update Interval Days' has the most significant impact. When the 'Update Interval Days' are lower (closer to 0), the average 'In App Purchases' is also lower, at around 0.16 to 0.32. As 'Update Interval Days' increase, the average 'In App Purchases' generally show an upward trend, peaking at around 0.51 for an 'Update Interval Days' of 4.9. Additionally, the 'Rating' also has a notable effect on the increase in 'In-App Purchases.' When the rating increases, the purchases have lower figures, while with an increasing rating, the purchases increase. However, when comparing other features, such as price, install_avg, and content_Rating, they did not appear to significantly affect purchase.

<img src="graphs/rf_5.png" alt="Graph" width="500">










