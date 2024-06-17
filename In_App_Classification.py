#!/usr/bin/env python
# coding: utf-8

# ### Loding Libraries

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize


# ### Loading Data

# In[2]:


data = pd.read_csv('D:/Python/assignment/Google APP/Google-Playstore.csv')


# In[3]:


data = pd.read_table('/Users/teresa930206/Downloads/Google-Playstore.csv', sep=',', header=0)


# ### Undersampling

# In[23]:


data["In App Purchases"].value_counts()


# In[24]:


195309/2


# In[27]:


purchases = data[data["In App Purchases"] == True]
seed = 12345
purchases = purchases.sample(n = 97654, random_state = seed)
purchases.shape


# In[16]:


97654*2


# In[17]:


sampling = data[data["In App Purchases"] == False]
unpurchases = sampling.sample(n = 195308, random_state = seed)
unpurchases.shape


# In[29]:


df = pd.concat([purchases, unpurchases],ignore_index = True)
df = df.sample(frac = 1).reset_index(drop = True)
df.shape


# ### Checking Data

# In[11]:


df.columns


# In[30]:


df.head(5)


# In[31]:


df['In App Purchases'].value_counts()


# In[32]:


column_drop = ['App Id', 'Developer Website','Developer Email', 'Privacy Policy', 'Scraped Time', 'App Name', 'Developer Id']
df = df.drop(columns = column_drop)


# ### Missing Values

# In[33]:


df.isnull().sum()


# In[35]:


null = df[df.isnull().sum(axis=1) >= 3]
null.shape


# In[39]:


df1 = df.drop(null.index)
df1.isnull().sum()


# In[40]:


df1 = df1.dropna(subset = ["Currency"])
df1.isnull().sum()


# In[41]:


df1.info()


# In[42]:


df1.describe()


# # Data Conversation (Date)

# In[43]:


df1_copy = df1.copy()
df1_copy.loc[:, 'Install_Avg'] = (df1['Minimum Installs'] + df1['Maximum Installs'])/2
df1_copy.drop(['Installs'], axis =1, inplace=True)


# In[44]:


df1_copy['Released'].head(2)


# In[45]:


# datatime transform
d_format = "%b %d, %Y"
df1_copy['Released'] = df1_copy['Released'].apply(lambda x: datetime.strptime(x, d_format) if isinstance(x, str) else x)
df1_copy['Last Updated'] = df1_copy['Last Updated'].apply(lambda x: datetime.strptime(x, d_format) if isinstance(x, str) else x)
df1_copy['Last Updated'].head(2)


# In[46]:


# calculate update interval dates
def cal_interval(row):
    year = 2021
    month = 6 
    dataset_date = datetime(year, month, 1)

    if row['Last Updated'] > row['Released']:
        return dataset_date - row['Last Updated']
    else: 
        return dataset_date - row['Released']


# In[47]:


# create column 'update interval days'
df1_copy['Update Interval Days'] = df1_copy.apply(cal_interval, axis =1)
df1_copy['Update Interval Days'] = df1_copy['Update Interval Days'].dt.days
df1_copy['Update Interval Days'].head()


# In[48]:


df1_copy.drop(['Last Updated'], axis =1, inplace=True)


# In[49]:


df1_copy.drop(['Released'], axis =1, inplace=True)


# In[50]:


df1_copy.head(5)


# In[51]:


df1_copy.isnull().sum()


# In[52]:


mean = df1_copy['Rating'].mean()
df1_copy['Rating'].fillna(mean, inplace = True)
df1_copy['Rating'] = round(df1_copy['Rating'], 1)


# In[53]:


mean = df1_copy['Rating Count'].mean()
df1_copy['Rating Count'].fillna(mean, inplace = True)
df1_copy['Rating Count'] = round(df1_copy['Rating'], 1)


# In[54]:


mean = df1_copy['Update Interval Days'].mean()
df1_copy['Update Interval Days'].fillna(mean, inplace = True)
df1_copy['Update Interval Days'] = round(df1_copy['Rating'], 1)


# In[55]:


df1_copy.isnull().sum()


# In[31]:


df1_copy.to_csv('df1.csv')


# # Data Conversation (encode)

# In[32]:


#df1 = pd.read_csv('D:/Python/assignment/Google APP/df_clean.csv')


# In[57]:


df1_1 = df1_copy


# In[58]:


bool_columns = ['Free', 'Ad Supported', 'In App Purchases', 'Editors Choice']
for col in bool_columns:
    df1_1[col] = df1_1[col].astype(int)


# In[59]:


label_encoder = LabelEncoder()


# In[60]:


df1_1['Content Rating'] = label_encoder.fit_transform(df1['Content Rating'])
class_labels = label_encoder.classes_
content_rating = df1_1['Content Rating']

for label, content_rating in enumerate(class_labels):
    print(f"Encoded integer {label} corresponds to category: {content_rating}")


# In[61]:


df1_1['Currency'] = label_encoder.fit_transform(df1['Currency'])
class_labels = label_encoder.classes_
currency = df1_1['Currency']

for label, currency in enumerate(class_labels):
    print(f"Encoded integer {label} corresponds to category: {currency}")


# In[62]:


df1_1['Category'] = label_encoder.fit_transform(df1['Category'])
class_labels = label_encoder.classes_
catagory = df1_1['Category']

for label, catagory in enumerate(class_labels):
    print(f"Encoded integer {label} corresponds to category: {catagory}")


# # Convert Minimum Android to Min Version

# In[63]:


# Define a function to extract the minimum version
def extract_min_version(version_str):
    # Use regular expression to find version numbers
    version_match = ''
    if isinstance(version_str, str):
        version_match = re.search(r'\d+(\.\d+)+', version_str)
    
    if version_match and isinstance(version_str, str):
        return version_match.group(0)
    else:# Handle 'Varies with device' by assigning it as 'None'
        return None

# Apply the function to the 'Minimum Android' column to extract minimum versions as strings
df1_1['Min Version'] = df1_1['Minimum Android'].apply(extract_min_version)


# In[64]:


df1_1['Min Version'].isnull().sum()


# In[65]:


import re

def extract_first_number(data):
    match = re.search(r'(\d+(\.\d+)?)', data)
    if match:
        return match.group(1)
    else:
        return None


# In[66]:


df1_1['Min Version'] = df1_1['Min Version'].apply(lambda x: extract_first_number(str(x))).astype(float)

# use mean to replace null
mean = df1_1['Min Version'].mean()
df1_1['Min Version'].fillna(mean, inplace = True)
df1_1['Min Version'] = round(df1_1['Min Version'], 1)


# In[67]:


df1_1['Min Version'].isnull().sum()


# In[68]:


# drop 'Minimum Android'
df1_1.drop('Minimum Android', axis=1, inplace=True)


# # Convert Size to Size_Numeric

# In[69]:


df1_1['Size'].unique()


# In[70]:


# Define a function to convert 'size' values to numerical values
def convert_size_to_numeric(size_str):
    # Dictionary to map suffixes to multipliers
    suffixes = {'K': 1, 'M': 1024, 'G': 1048576}

    # Check if the value contains a valid suffix
    for suffix, multiplier in suffixes.items():
        if size_str.endswith(suffix):
            # Extract the numeric part and convert it to a float
            numeric_part = float(size_str[:-1 * len(suffix)])
            return numeric_part * multiplier

    return None  # Handle cases like 'Varies with device'

# Convert 'size' values to numerical values
df1_1['size_numeric(KB)'] = df1_1['Size'].apply(convert_size_to_numeric)

# Calculate the mean of numerical 'size' values, excluding 'Varies with device'
mean_size = df1_1[df1_1['size_numeric(KB)'].notnull()]['size_numeric(KB)'].mean()

# Replace 'Varies with device' with the calculated mean
df1_1['size_numeric(KB)'].fillna(mean_size, inplace=True)

# Print the DataFrame with processed data
print(df1_1[['Size', 'size_numeric(KB)']])


# In[71]:


# drop 'Size'
df1_1.drop('Size', axis=1, inplace=True)


# In[91]:


df1_1.head(5)


# In[73]:


df1_1.isnull().sum()


# In[50]:


df1_1.to_csv('df_modeling.csv')


# ### Outlier

# In[97]:


df1_2 = df1_1.copy()


# In[98]:


df1_2.head(3)


# In[99]:


df1_2.drop(['Minimum Installs', 'Maximum Installs'], axis=1, inplace=True)


# In[100]:


df1_2.columns


# https://towardsdatascience.com/creating-boxplots-of-well-log-data-using-matplotlib-in-python-34c3816e73f4

# In[101]:


numv_list = ['Rating', 'Rating Count','Price', 'Install_Avg', 'Update Interval Days', 'size_numeric(KB)'] 


# In[102]:


def box_plots(s):
    print("[", s, "]")
    q1 = df1_2[s].quantile(0.25)
    q3 = df1_2[s].quantile(0.75)
    iqr = q3 - q1    
    print('q1: ', q1, 'q3: ', q3, "\n")
 
    lower_outlier = df1_2[df1_2[s] < q1 - 1.5 * iqr][s]
    upper_outlier = df1_2[df1_2[s] > q3 + 1.5 * iqr][s]
    print('(lower_outlier)' , len(lower_outlier))
    print(lower_outlier, "\n")
    print('(upper_outlier)', len(upper_outlier))
    print(upper_outlier, "\n\n")

    red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')
    plt.boxplot(df1_2[s], flierprops=red_circle, vert=False)
    plt.title(s)
    plt.tick_params(axis='y', labelsize=14)
    plt.show()


# In[103]:


for s in numv_list:
    box_plots(s)


# # Fix the outlier 

# In[105]:


df1_3 = df1_2.copy()
df1_3.head(5)


# ### <b> <font color ='green'> Price : 
# Because 0 accounts for 98% of the total data, the upper bound tends to be 0, making the paid data appear
# anomalous. We think this skewed fits the real world and we don't use price as a target variable. Therefore, I think we can leave the outlier here. 

# ### <b> <font color ='green'> Install_Avg: delete the extrem value 

# In[106]:


df_t = df1_3.sort_values(by = 'Install_Avg', ascending = False)
df_t['Install_Avg'].head()


# In[107]:


# Delete data beyong 5 billion
df1_3 = df1_3[df1_3['Install_Avg'] <= 5000000000]
df1_3['Install_Avg'].head()


# In[108]:


df_t = df1_3.sort_values(by = 'Install_Avg', ascending = False)
df_t['Install_Avg'].head()


# ### <b> <font color ='green'> size_numeric(KB):
# The outlier data actually aligns with real-world situations, and its maximum value is not outrageous. It contains valid information that cannot be easily deleted. Moreover, the number of outliers is not large compared to our overall data, so we are considering not handling them.

# In[109]:


numv_list = ['Rating', 'Rating Count','Price', 'Install_Avg', 'Update Interval Days', 'size_numeric(KB)']


# ### Plot IQR box plot

# In[110]:


def box_plots(s):
    print("[", s, "]")
    q1 = df1_3[s].quantile(0.25)
    q3 = df1_3[s].quantile(0.75)
    iqr = q3 - q1    
    print('q1: ', q1, 'q3: ', q3, "\n")
 
    lower_outlier = df1_3[df1_3[s] < q1 - 1.5 * iqr][s]
    upper_outlier = df1_3[df1_3[s] > q3 + 1.5 * iqr][s]
    print('(lower_outlier)' , len(lower_outlier))
    print(lower_outlier, "\n")
    print('(upper_outlier)', len(upper_outlier))
    print(upper_outlier, "\n\n")

    red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')
    plt.boxplot(df1_3[s], flierprops=red_circle, vert=False)
    plt.title(s)
    plt.tick_params(axis='y', labelsize=14)
    plt.show()


# In[86]:


for s in numv_list:
    box_plots(s)


# ### <b> <font color ='green'> For outlier:
# we only process the outliers in "Install_Avg" and retain outliers in other fields to accurately represent real-world app store installation situations.

# In[139]:


df1_3.to_csv('df_modeling.csv')


# # EDA

# In[116]:


df_f = df1_3.copy()


# In[117]:


df_f.head(5)


# In[118]:


result_installA = df_f.groupby('In App Purchases')['Install_Avg'].mean()
result_installA


# In[119]:


purchase_0_data = df_f[df_f['In App Purchases'] == 0]
purchase_1_data = df_f[df_f['In App Purchases'] == 1]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(purchase_0_data.index, purchase_0_data['Install_Avg'], label='Purchase 0')
plt.xlabel('Data Point Index')
plt.ylabel('Install_Avg')
plt.title('Install Numbers for Purchase 0')

plt.subplot(1, 2, 2)
plt.scatter(purchase_1_data.index, purchase_1_data['Install_Avg'], label='Purchase 1', color='orange')
plt.xlabel('Data Point Index')
plt.ylabel('Install_Avg')
plt.title('Install Numbers for Purchase 1')

plt.tight_layout()
plt.show()


# In[120]:


result_siseA = df_f.groupby('In App Purchases')['size_numeric(KB)'].mean()


# In[121]:


result_siseA


# In[122]:


purchase_0_data = df_f[df_f['In App Purchases'] == 0]
purchase_1_data = df_f[df_f['In App Purchases'] == 1]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(purchase_0_data.index, purchase_0_data['size_numeric(KB)'], label='Purchase 0')
plt.xlabel('Data Point Index')
plt.ylabel('size_numeric(KB)')
plt.title('Install Numbers for Purchase 0')

plt.subplot(1, 2, 2)
plt.scatter(purchase_1_data.index, purchase_1_data['size_numeric(KB)'], label='Purchase 1', color='orange')
plt.xlabel('Data Point Index')
plt.ylabel('size_numeric(KB)')
plt.title('Install Numbers for Purchase 1')

plt.tight_layout()
plt.show()


# In[123]:


result_RatingC = df_f.groupby('In App Purchases')['Rating Count'].mean()
result_RatingC 


# In[124]:


purchase_1_data = df_f[df_f['In App Purchases'] == 1]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(purchase_0_data.index, purchase_0_data['Rating Count'], label='Purchase 0')
plt.xlabel('Data Point Index')
plt.ylabel('Rating Count')
plt.title('Install Numbers for Purchase 0')

plt.subplot(1, 2, 2)
plt.scatter(purchase_1_data.index, purchase_1_data['Rating Count'], label='Purchase 1', color='orange')
plt.xlabel('Data Point Index')
plt.ylabel('Rating Count')
plt.title('Install Numbers for Purchase 1')

plt.tight_layout()
plt.show()


# In[125]:


result_Rating = df_f.groupby('In App Purchases')['Rating'].mean()
result_Rating


# In[126]:


purchase_1_data = df_f[df_f['In App Purchases'] == 1]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(purchase_0_data.index, purchase_0_data['Rating'], label='Purchase 0')
plt.xlabel('Data Point Index')
plt.ylabel('Rating')
plt.title('Install Numbers for Purchase 0')

plt.subplot(1, 2, 2)
plt.scatter(purchase_1_data.index, purchase_1_data['Rating'], label='Purchase 1', color='orange')
plt.xlabel('Data Point Index')
plt.ylabel('Rating')
plt.title('Install Numbers for Purchase 1')

plt.tight_layout()
plt.show()


# In[127]:


result_Rating = df_f.groupby('In App Purchases')['Min Version'].mean()
result_Rating


# In[128]:


bar_width =0.8
crosstab_0 = pd.crosstab( purchase_0_data['Min Version'], purchase_0_data['In App Purchases'])
ax = crosstab_0.plot(kind='bar', width=bar_width, cmap='inferno')
plt.xlabel('Purchase 0')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.title('Purchase 0 vs. Min Version')
ax.legend(title='Version', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

crosstab_1 = pd.crosstab(purchase_1_data['Min Version'], purchase_1_data['In App Purchases'])
ax = crosstab_1.plot(kind='bar', width=bar_width, cmap='viridis')
plt.xlabel('Purchase 1')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.title('Purchase 1 vs. Min Version')

ax.legend(title='Version', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()


# In[129]:


contingency_table = pd.crosstab(df_f['Ad Supported'], df_f['In App Purchases'])

plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', cbar=False)

plt.xlabel('In App Purchases')
plt.ylabel('Ad Supported')
plt.title('Relationship between In App Purchases and Ad Supported')

plt.show()


# In[135]:


print(26271/ (100697+26271))
print(70148/ (92755+70148))
print((92755+70148) / (100697+26271+92755+70148))


# In[134]:


contingency_table = pd.crosstab(df_f['Editors Choice'], df_f['In App Purchases'])

plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='copper', cbar=False)

plt.xlabel('In App Purchases')
plt.ylabel('Editors Choice')
plt.title('Relationship between In App Purchases and Editors Choice')

plt.show()


# In[136]:


print(96099/ (193433+96099))
print(320/ ((19+320)))
print((19+320) / (19+320+193433+96099))


# # Correlation Analysis

# In[140]:


corr = df_f.corr()


# In[141]:


ax = sns.heatmap(
    corr, 
    vmin = -1, vmax = 1, center = 0,
    cmap = sns.diverging_palette(20, 220, n = 200),
    square = True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 90,
    horizontalalignment = 'right'
);
ax.set_title('Correlation Heat Map')


# # Split the dataset

# In[6]:


df_code = pd.read_csv('D:/Python/assignment/Google APP/df_modeling.csv')


# In[3]:


df_code = pd.read_csv('df_modeling.csv')


# In[4]:


df_code.head(5)


# In[5]:


X = df_code.drop("In App Purchases", axis=1)
y = df_code["In App Purchases"]


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 12, test_size = 0.3,stratify = y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Random Forest (Miryang Kim)

# In[58]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_nor = scaler.fit_transform(X_train)

X_test_nor = scaler.transform(X_test)


# In[59]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
param = {'n_estimators':[50, 100, 500], 'max_depth' : [5,10, 20], 'min_samples_split': [2,4,8], 
         'min_samples_leaf':[1,3,5], 'max_samples': [0.1,0.2,0.3]}


# In[107]:


RFC_model_RS =  RandomizedSearchCV(estimator= RandomForestClassifier(), param_distributions=param, n_iter=100, cv=5, 
                                   scoring='precision', verbose=3, n_jobs=-1, return_train_score=True)


# In[108]:


RFC_model_RS.fit(X_train_nor, y_train)
best_params_RFC = RFC_model_RS.best_params_


# In[109]:


best_params_RFC


# In[111]:


RFC_model_RS_results = RFC_model_RS.cv_results_


# In[131]:


RFC_model_basic = RandomForestClassifier(max_depth = 500, max_samples = 0.3, min_samples_leaf = 5, min_samples_split = 2, 
                                         n_estimators = 20,random_state=12)


# In[132]:


import time
start_time =  time.time()
RFC_model_basic.fit(X_train_nor, y_train)
end_time = time.time()


# In[133]:


train_time = end_time - start_time
train_time


# In[134]:


pred_RFC_basic = RFC_model_basic.predict(X_test_nor)


# In[135]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score, precision_score, roc_curve, auc
confusion_matrix(y_test, pred_RFC_basic)
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, roc_curve, auc, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


# In[136]:


def model_report(model, X_test, y_test):
    predition = model.predict(X_test)
    
    confusion = confusion_matrix(y_test, predition)
    confusion_display = ConfusionMatrixDisplay(confusion)
    confusion_display.plot(cmap='pink_r')
    
    print(classification_report(y_test, predition))
    
    recall = recall_score(y_test, predition)
    print("Recall (Sensitivity, TPR):", recall)
    
    precision = precision_score(y_test, predition)
    print("Precision (PPV):", precision)
    
    FPR = 1 - recall
    print("FPR (1 - Sensitivity):", FPR)
    
    proba = model.predict_proba(X_test)
    proba_1 = proba[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, proba_1)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize = (7, 5))
    plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
    
    prec, recall, _ = precision_recall_curve(y_test, proba_1)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    plt.grid(True, linestyle='--', linewidth=0.5, color='black')
    plt.title("Precision-Recall Curve")
    plt.show()


# In[137]:


model_report(RFC_model_basic, X_test_nor, y_test)


# # Class_weight

# In[138]:


from sklearn.utils.class_weight import compute_class_weight

class_labels = np.unique(y_train)
print(class_labels)
sklearn_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
print(sklearn_weights)
class_weight_dict = dict(zip(class_labels, sklearn_weights))
print(class_weight_dict)


# In[139]:


RFC_mode_classweight = RandomForestClassifier(max_depth = 500, max_samples = 0.3, min_samples_leaf = 5, min_samples_split = 2, 
                                         n_estimators = 20,class_weight = class_weight_dict, 
                                              random_state=12)


# In[140]:


import time
start_time =  time.time()
RFC_mode_classweight.fit(X_train_nor, y_train)
end_time = time.time()


# In[141]:


model_report(RFC_mode_classweight, X_test_nor, y_test)


# # SMOTE

# In[142]:


from imblearn.over_sampling import SMOTE
count_0 = np.count_nonzero(y_train == 0)
count_1 = np.count_nonzero(y_train == 1)
print(count_0)
print(count_1)
len(y_train)


# In[143]:


oversample = SMOTE()
X_res, y_res = oversample.fit_resample(X_train_nor, y_train)


# In[144]:


count_0 = np.count_nonzero(y_res == 0)
count_1 = np.count_nonzero(y_res == 1)
print(count_0)
print(count_1)
len(y_train)


# In[145]:


RFC_model_smote = RandomForestClassifier(max_depth = 500, max_samples = 0.3, min_samples_leaf = 5, min_samples_split = 2, 
                                         n_estimators = 20, random_state=12)


# In[146]:


import time
start_time =  time.time()
RFC_model_smote.fit(X_res, y_res)
end_time = time.time()


# In[147]:


model_report(RFC_model_smote, X_test_nor, y_test)


# In[148]:


f_importance = RFC_mode_classweight.feature_importances_
f_labels = X_test.columns


# In[149]:


feature_importance_pairs = list(zip(f_labels, f_importance))
feature_importance_pairs = pd.DataFrame(feature_importance_pairs)
f_imp = feature_importance_pairs.sort_values(by = 1, ascending=False)


# In[150]:


import matplotlib.pyplot as plt
plt.bar(f_imp[0], f_imp[1])
plt.xticks(rotation=70);


# In[ ]:





# # Decision Tree Classifier (Xiaotong Huang)

# In[151]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV


# In[154]:


dt_classifier = DecisionTreeClassifier()

param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None] + list(np.arange(5, 16, 1)),
    'min_samples_split': list(np.arange(2, 11, 1)),
    'min_samples_leaf': list(np.arange(1, 5, 1))
}

random_search = RandomizedSearchCV(estimator = dt_classifier, 
                                   param_distributions = param_dist, 
                                   n_iter = 50, scoring = 'precision', cv = 5, random_state = 42)


# In[155]:


random_search.fit(X_train, y_train)


# In[156]:


best_params = random_search.best_params_
print("Optimal Hyperparameter Combination：", best_params)


# In[157]:


tree = random_search.best_estimator_


# In[158]:


model_report(tree, X_test, y_test)


# In[159]:


importances = tree.feature_importances_
feature_names = X_train.columns


# In[160]:


feature_importance = list(zip(feature_names, importances))
# Ranking the importance of features
feature_importance.sort(key=lambda x: x[1], reverse = True)
# Extract feature names and importance values
sorted_feature_names, sorted_importances = zip(*feature_importance)


# In[161]:


plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
plt.yticks(range(len(sorted_importances)), sorted_feature_names)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Plot')
plt.show()


# # Class_weight

# In[162]:


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_labels = np.unique(y_train)
print(class_labels)
sklearn_weights = compute_class_weight(class_weight = 'balanced', classes = class_labels, y = y_train)
print(sklearn_weights)
class_weight_dict = dict(zip(class_labels, sklearn_weights))
print(class_weight_dict)


# In[166]:


tree_classweight = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 4, min_samples_split = 6,
                                             class_weight = class_weight_dict, criterion = 'gini')
tree_classweight.fit(X_train, y_train)


# In[167]:


model_report(tree_classweight, X_test, y_test)


# # SMOTE

# In[50]:


get_ipython().system('pip install imbalanced-learn')


# In[168]:


from imblearn.over_sampling import SMOTE
count_0 = np.count_nonzero(y_train == 0)
count_1 = np.count_nonzero(y_train == 1)
print(count_0)
print(count_1)
len(y_train)


# In[169]:


oversample = SMOTE()
X_res, y_res = oversample.fit_resample(X_train, y_train)


# In[170]:


count_0 = np.count_nonzero(y_res == 0)
count_1 = np.count_nonzero(y_res == 1)
print(count_0)
print(count_1)
len(y_train)


# In[171]:


tree_smote = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 4, min_samples_split = 6, criterion = 'gini')
tree_smote.fit(X_res, y_res)


# In[172]:


model_report(tree_smote, X_test, y_test)


# In[ ]:





# # Gradient Boosting Classifier (Yi-Wen Shen)

# In[173]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_nor = scaler.fit_transform(X_train)

X_test_nor = scaler.transform(X_test)


# In[174]:


# Create an instance of the GradientBoostingClassifier
classifier = GradientBoostingClassifier()


# In[175]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
# Create a dictionary containing the hyperparameters and their respective distributions:
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(1, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],

}


# In[192]:


# Perform randomized search:
random_search = RandomizedSearchCV(
    classifier, param_distributions=param_dist, n_iter=10, cv=5, scoring ='precision', n_jobs=-1, random_state=42
)
random_search.fit(X_train_nor, y_train)


# In[193]:


# Print the best hyperparameters and the corresponding score:
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)


# In[194]:


best_gbm_model = random_search.best_estimator_ 


# In[195]:


model_report(best_gbm_model, X_test_nor, y_test)


# In[196]:


# Get feature importances for the best model
feature_importance = best_gbm_model.feature_importances_

# Associate feature importances with feature names or column names
feature_names = X_train.columns  # Assuming X_train is a DataFrame
feature_importance_dict = dict(zip(feature_names, feature_importance))

# Sort features by importance (optional)
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print or visualize the feature importance
for feature, importance in sorted_feature_importance:
    print(f"Feature: {feature}, Importance: {importance}")


# In[197]:


# Extract feature names and importances
feature_names = [feature for feature, _ in sorted_feature_importance]
importance_values = [value for _, value in sorted_feature_importance]

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_names)), importance_values, align='center')
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Gradient Boosting Classifier')
plt.show()


# # Class Weight

# In[198]:


from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

class_labels = np.unique(y_train)
print(class_labels)
sklearn_weights = compute_sample_weight(class_weight='balanced', y=y_train)
print(sklearn_weights)
sample_weight_dict = dict(zip(class_labels, sklearn_weights))
print(sample_weight_dict)


# In[199]:


clf_weight = GradientBoostingClassifier(learning_rate = 0.2, max_depth = 6, min_samples_leaf = 18, min_samples_split = 13, 
                                        n_estimators = 63, subsample = 0.9, random_state=42)


# In[200]:


import time
start_time =  time.time()
clf_weight.fit(X_train, y_train, sample_weight= sklearn_weights)
end_time = time.time()
running_time = end_time - start_time

print(f"Class Weight took {running_time:.2f} seconds to complete.")


# In[201]:


model_report(clf_weight, X_test, y_test)


# # SMOTE

# In[202]:


oversample = SMOTE()
X_res, y_res = oversample.fit_resample(X_train, y_train)


# In[204]:


clf_SMOTE = GradientBoostingClassifier(learning_rate = 0.2, max_depth = 6, min_samples_leaf = 18, min_samples_split = 13, 
                                        n_estimators = 63, subsample = 0.9, random_state=42)


# In[205]:


import time
start_time =  time.time()
clf_SMOTE.fit(X_res, y_res)
end_time = time.time()
running_time = end_time - start_time

print(f"SMOTE took {running_time:.2f} seconds to complete.")


# In[206]:


model_report(clf_SMOTE, X_test, y_test)


# In[ ]:




