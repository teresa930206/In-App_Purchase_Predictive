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

We applied ```LabelEncoder() from sklearn to encode the categorical values.
<img src="graphs/data_encode.png" alt="Graph" width="500">


