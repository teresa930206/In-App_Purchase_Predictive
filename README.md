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



