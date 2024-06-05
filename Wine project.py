#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')


# In[4]:


wine = pd.read_csv("C:/Users/shrey/Downloads/winequality-red.csv")
print("Successfully Imported Data!")
wine.head()


# In[5]:


print(wine.shape)


# In[6]:


wine.describe(include='all')


# In[7]:


print(wine.isna().sum())


# In[8]:


wine.corr()


# In[9]:


wine.groupby('quality').mean()


# In[13]:


sns.countplot(x='quality', data=wine)
plt.show()


# In[24]:


rainbow_palette = sns.color_palette("hsv", n_colors=len(wine['pH'].unique()))
sns.countplot(x='pH', data=wine, palette=rainbow_palette)
plt.show()


# In[23]:


rainbow_palette = sns.color_palette("hsv", n_colors=len(wine['alcohol'].unique()))
sns.countplot(x='alcohol', data=wine, palette=rainbow_palette)
plt.show()


# In[22]:


rainbow_palette = sns.color_palette("hsv", n_colors=len(wine['fixed acidity'].unique()))
sns.countplot(x='fixed acidity', data=wine, palette=rainbow_palette)
plt.show()


# In[25]:


rainbow_palette = sns.color_palette("hsv", n_colors=len(wine['volatile acidity'].unique()))
sns.countplot(x='volatile acidity', data=wine, palette=rainbow_palette)
plt.show()


# In[26]:


rainbow_palette = sns.color_palette("hsv", n_colors=len(wine['citric acid'].unique()))
sns.countplot(x='citric acid', data=wine, palette=rainbow_palette)
plt.show()


# In[27]:


rainbow_palette = sns.color_palette("hsv", n_colors=len(wine['density'].unique()))
sns.countplot(x='density', data=wine, palette=rainbow_palette)
plt.show()


# In[28]:


sns.kdeplot(wine.query('quality > 2').quality)


# In[29]:


sns.distplot(wine['alcohol'])


# In[30]:


wine.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)


# In[33]:


wine.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)


# In[34]:


wine.hist(figsize=(10,10),bins=50)
plt.show()


# In[40]:


numeric_columns = wine.select_dtypes(include=['float64', 'int64'])
corr = numeric_columns.corr()

# Create a heatmap with annotations
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)

# Display the plot
plt.show()


# In[42]:


sns.pairplot(wine)


# In[43]:


sns.violinplot(x='quality', y='alcohol', data=wine)


# In[44]:


wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]# Separate feature variables and target variable
X = wine.drop(['quality','goodquality'], axis = 1)
Y = wine['goodquality']


# In[45]:


wine['goodquality'].value_counts()


# In[46]:


X


# In[47]:


print(Y)


# In[52]:


X_numeric = X.select_dtypes(include=['float64', 'int64'])

classifiern = ExtraTreesClassifier()
classifiern.fit(X_numeric, Y)

# Get feature importances
feature_importances = classifiern.feature_importances_

print("Feature Importances:")
for feature, importance in zip(X_numeric.columns, feature_importances):
    print(f"{feature}: {importance}")


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)


# In[56]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Assuming you have loaded your data and split it into training and testing sets
# X_train, Y_train, X_test, Y_test = ...

# Select only numeric columns for training and testing sets
X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train_numeric, Y_train)

# Make predictions on the test data
Y_pred = model.predict(X_test_numeric)

# Calculate and print the accuracy score
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy Score:", accuracy)

# Display the confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", conf_matrix)


# In[57]:


confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)


# In[59]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Assuming you have loaded your data and split it into training and testing sets
# X_train, Y_train, X_test, Y_test = ...

# Select only numeric columns for training and testing sets
X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])

# Create a KNN classifier with 3 neighbors
model = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training data
model.fit(X_train_numeric, Y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_numeric)

# Calculate and print the accuracy score
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy Score:", accuracy)


# In[61]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Assuming you have loaded your data and split it into training and testing sets
# X_train, Y_train, X_test, Y_test = ...

# Select only numeric columns for training and testing sets
X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])

# Create an SVM classifier
model = SVC()

# Train the model on the training data
model.fit(X_train_numeric, Y_train)

# Make predictions on the test data
pred_y = model.predict(X_test_numeric)

# Calculate and print the accuracy score
accuracy = accuracy_score(Y_test, pred_y)
print("Accuracy Score:", accuracy)


# In[63]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Assuming you have loaded your data and split it into training and testing sets
# X_train, Y_train, X_test, Y_test = ...

# Select only numeric columns for training and testing sets
X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])

# Create a Decision Tree classifier with entropy criterion
model = DecisionTreeClassifier(criterion='entropy', random_state=7)

# Train the model on the training data
model.fit(X_train_numeric, Y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_numeric)

# Calculate and print the accuracy score
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy Score:", accuracy)


# In[65]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Assuming you have loaded your data and split it into training and testing sets
# X_train, Y_train, X_test, Y_test = ...

# Select only numeric columns for training and testing sets
X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])

# Create a RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)

# Train the model on the training data
model2.fit(X_train_numeric, Y_train)

# Make predictions on the test data
y_pred2 = model2.predict(X_test_numeric)

# Calculate and print the accuracy score
accuracy2 = accuracy_score(Y_test, y_pred2)
print("Accuracy Score:", accuracy2)


# In[66]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

# Assuming you have loaded your data and split it into training and testing sets
# X_train, Y_train, X_test, Y_test = ...

# Select only numeric columns for training and testing sets
X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])

# Create a Gaussian Naive Bayes classifier
model3 = GaussianNB()

# Train the model on the training data
model3.fit(X_train_numeric, Y_train)

# Make predictions on the test data
y_pred3 = model3.predict(X_test_numeric)

# Calculate and print the accuracy score
accuracy3 = accuracy_score(Y_test, y_pred3)
print("Accuracy Score:", accuracy3)


# In[13]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','KNN', 'SVC','Decision Tree' ,'GaussianNB','Random Forest','Xgboost'],
    'Score': [0.870,0.872,0.868,0.864,0.833,0.893,0.879]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df


# In[ ]:


C:/Users/shrey/Downloads/winequality-red.csv"

