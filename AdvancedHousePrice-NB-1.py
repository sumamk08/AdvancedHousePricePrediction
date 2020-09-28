#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew 
from scipy.stats import norm
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from datetime import datetime


# In[2]:


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))


# In[3]:


# Load the data
data = pd.read_csv('train.csv')


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


# Drop the ID column as its not required for prediction

train_id = data['Id']
data  =  data.drop('Id', axis = 'columns')


# In[7]:


data.shape


# ###### Missing Data

# In[8]:


missing_df = data.isnull().mean() * 100
missing_df = missing_df.drop(missing_df[missing_df == 0].index).sort_values(ascending = False)
missing_df = pd.DataFrame(missing_df, columns = {'Missing Ratio'})
missing_df


# In[9]:


plt.rcParams['figure.figsize'] = (20, 10)
sns.barplot(x = missing_df.index, y = missing_df['Missing Ratio'])
plt.title(' Missing Ratio')
plt.xlabel(' Features ')
plt.ylabel(' Missing Percentage ')
plt.xticks(rotation = '45')


# #### We start handling the missing values by going through each features

# 1. PoolQC - Pool Quality in which NA means the house doesnt contain pool which makes sense of having so many missing (99.5%)

# In[10]:


data['PoolQC'] = data['PoolQC'].fillna("None")


# 2. MiscFeature - Containts details of Garage, Tennis Court, Shed and other details, so we can set to None as the home doesn't contain these facilities

# In[11]:


data['MiscFeature'].unique()


# In[12]:


data['MiscFeature'] = data['MiscFeature'].fillna('None')


# 3. Alley - Type of alley access to home (Paved, Gavel or no access)

# In[13]:


data['Alley'].unique()


# In[14]:


data['Alley'] = data['Alley'].fillna('None')


# 4. Fence - Quality of fence Good, Min Privacy, Good wood or No Fence

# In[15]:


data['Fence'].unique()


# In[16]:


data['Fence'] = data['Fence'].fillna('None')


# 5. FireplaceQu - Quality of Fire Place (None for no fire place)

# In[17]:


data['FireplaceQu'].unique()


# In[18]:


data['FireplaceQu'] = data['FireplaceQu'].fillna('None')


# 6. LotFrontage: Linear feet of street connected to property - since the distance for house property might be mostly similiar to neigborhood we can use Median for the missing ones

# In[19]:


data['LotFrontage'].isnull().sum()


# In[20]:


data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# 7. GarageYrBlt: Year garage was built

# In[21]:


data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)


# 8. GarageType - Type of Garage, NA for no garage

# In[22]:


data['GarageType'].unique()


# In[23]:


data['GarageType'] = data['GarageType'].fillna('None')


# 9. GarageFinish - Interior finish of the garage; NA for no garage

# In[24]:


data['GarageFinish'] = data['GarageFinish'].fillna('None')


# 10. GarageQual - Garage quality; NA for no garage

# In[25]:


data['GarageQual'] = data['GarageQual'].fillna('None')


# 11. GarageCond: Garage condition; NA for no garage

# In[26]:


data['GarageCond'] = data['GarageCond'].fillna('None')


# 12. BsmtFinType2: Rating of basement finished area (if multiple types); NA no basement

# In[27]:


data['BsmtFinType2'] = data['BsmtFinType2'].fillna('None')


# 13. BsmtExposure: Refers to walkout or garden level walls; NA for no basement

# In[28]:


data['BsmtExposure'] = data['BsmtExposure'].fillna('None')


# 14. BsmtFinType1: Rating of basement finished area; NA for no basement

# In[29]:


data['BsmtFinType1'] = data['BsmtFinType1'].fillna('None')


# 15. BsmtCond: Evaluates the general condition of the basement; NA for no basement

# In[30]:


data['BsmtCond'] = data['BsmtCond'].fillna('None')


# 16. BsmtQual: Evaluates the height of the basement; NA for no basement

# In[31]:


data['BsmtQual'] = data['BsmtQual'].fillna('None')


# 17. MasVnrArea: Masonry veneer area in square feet; NA means no Masonry Veneer for the house so we fill with 0

# In[32]:


data['MasVnrArea'] = data['MasVnrArea'].fillna(0)


# 18. MasVnrType: Masonry veneer type; NA for none

# In[33]:


data['MasVnrType'].unique()


# In[34]:


data['MasVnrType'] = data['MasVnrType'].fillna('None')


# 19. Electrical: Electrical system; Since only one missing value we replace with mode (SBrkr)

# In[35]:


data['Electrical'].isnull().sum()


# In[36]:


data['Electrical'].value_counts()


# In[37]:


data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])


# In[38]:


data['Utilities'].value_counts()


# Since all the values point to All Pub and only one to NoSeWa, it wont help the model so we can safely drop this column

# In[39]:


data.drop(['Utilities'], axis = 'columns', inplace = True)


# In[40]:


data.isnull().sum().sum()


# No Missing value present

# Outliers 

# In[41]:


# Quantitative features
quantitative = [f for f in data.columns if data.dtypes[f] != 'object']
quantitative.remove('SalePrice') # Remove the Target Feature
qualitative = [f for f in data.columns if data.dtypes[f] == 'object']


# In[42]:


print('Quantitative count: ', len(quantitative))
print('Qualitative count: ', len(qualitative))


# In[43]:


plt.rcParams['figure.figsize'] = (20, 50)
list_features = ['MSSubClass', 'LotFrontage','LotArea','MasVnrArea',
                 'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
                 'GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
                 'ScreenPorch','MiscVal']

i = 0
fig, axs = plt.subplots(9, 2)
for row in range(0,9):
    for col in range(0,2):
        sns.scatterplot(x = list_features[i], y = 'SalePrice', data = data, ax = axs[row][col])
        i = i + 1


# In[44]:


plt.rcParams['figure.figsize'] = (6, 4)
sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = data)


# In[45]:


# Deleting the outlier

data = data.drop(data[(data['GrLivArea']>4000) & (data['SalePrice']<300000)].index)


# In[46]:


plt.rcParams['figure.figsize'] = (6, 4)
sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = data)


# In[47]:


skewed_features = data[quantitative].apply(lambda x: skew(x)).sort_values(ascending = False)
skewed_features[skewed_features > 0.75]


# In[48]:


skewed_features = skewed_features[skewed_features > 0.75]
skewed_features.index


# Checking the correlation of skewed data

# In[49]:


plt.rcParams['figure.figsize'] = (20, 10)
sns.heatmap(data[['MiscVal', 'PoolArea', 'LotArea', '3SsnPorch', 'LowQualFinSF',
       'KitchenAbvGr', 'BsmtFinSF2', 'ScreenPorch', 'BsmtHalfBath',
       'EnclosedPorch', 'MasVnrArea', 'OpenPorchSF', 'LotFrontage',
       'WoodDeckSF', 'MSSubClass', 'GrLivArea', 'BsmtUnfSF', '1stFlrSF',
       '2ndFlrSF', 'BsmtFinSF1','SalePrice']].corr(), annot = True)


# In[50]:


# We are dropping these features 'GrLivArea','1stFlrSF','MasVnrArea','BsmtFinSF1' from transformation as
# they are haivng high correlation with the target feature (Sales Price)

skewed_features = skewed_features.drop(['GrLivArea','1stFlrSF','MasVnrArea','BsmtFinSF1'])
for feature in skewed_features.index:
#     data[feature] = boxcox1p(data[feature], 0.15)
    data[feature] = np.sqrt(data[feature])
#     data[feature] = np.log1p(data[feature])
#     data[feature] = boxcox1p(data[feature], boxcox_normmax(data[feature] + 1))
            
    


# In[51]:


data[quantitative].apply(lambda x: skew(x)).sort_values(ascending = False)


# In[52]:


#Year and month sold are transformed into categorical features.
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)


# In[53]:



data['YrBltAndRemod']=data['YearBuilt']+data['YearRemodAdd']

data['TotalSF']=data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

data['Total_sqr_footage'] = (data['BsmtFinSF1'] + data['BsmtFinSF2'] +
                                 data['1stFlrSF'] + data['2ndFlrSF'])

data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +
                               data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))

data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +
                              data['EnclosedPorch'] + data['ScreenPorch'] +
                              data['WoodDeckSF'])


#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor

# In[54]:


# Ordinal Encoding

order_map = {'Ex': 1, 'Gd': 2, 'TA': 3, 'Fa': 4, 'Po': 5, 'None':6}
garage_finish_order = {'Fin': 1, 'RFn': 2, 'Unf': 3, 'None': 4}
lot_shape_order = { 'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3' : 4, 'None' : 5}
fence_order = {'GdPrv' : 1,'MnPrv' : 2,'GdWo':3, 'MnWw':4, 'None':5 }
BsmtExposure_order = { 'Gd' : 1, 'Av' : 2, 'Mn' : 3, 'No' : 4, 'None' : 5}
BsmtFinType_order = {'GLQ' : 1, 'ALQ' : 2, 'BLQ' : 3, 'Rec' : 4, 'LwQ': 5, 'Unf' : 6, 'None' : 7}
                                


# In[55]:


data['ExterQual'] = data['ExterQual'].map(order_map)
data['BsmtQual'] = data['BsmtQual'].map(order_map)
data['BsmtCond'] = data['BsmtCond'].map(order_map)
data['HeatingQC'] = data['HeatingQC'].map(order_map)
data['FireplaceQu'] = data['FireplaceQu'].map(order_map)
data['KitchenQual'] = data['KitchenQual'].map(order_map)
data['GarageQual'] = data['GarageQual'].map(order_map)
data['GarageCond'] = data['GarageCond'].map(order_map)
data['PoolQC'] = data['PoolQC'].map(order_map)

data['GarageFinish'] = data['GarageFinish'].map(order_map)
data['LotShape'] = data['LotShape'].map(lot_shape_order)
data['Fence'] = data['Fence'].map(fence_order)
data['BsmtExposure'] = data['BsmtExposure'].map(BsmtExposure_order)
data['BsmtFinType1'] = data['BsmtFinType1'].map(BsmtFinType_order)
data['BsmtFinType2'] = data['BsmtFinType2'].map(BsmtFinType_order)


# In[56]:


(mu, sigma) = norm.fit(data['SalePrice'])
sns.displot(data['SalePrice'], kde = True, bins = 30)
plt.ylabel('Frequency')
plt.title(' Sale Price Distribution ')
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')


# In[57]:


plt.rcParams['figure.figsize'] = (6,6)
stats.probplot(data['SalePrice'],plot = plt)


# In[58]:


# Log Transformation for Sales Price
data['SalePrice'] = np.log1p(data['SalePrice'])


# In[59]:


(mu, sigma) = norm.fit(data['SalePrice'])
sns.displot(data['SalePrice'], kde = True, bins = 30)
plt.ylabel('Frequency')
plt.title(' Sale Price Distribution ')
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')


# In[60]:


plt.rcParams['figure.figsize'] = (6,6)
stats.probplot(data['SalePrice'],plot = plt)


# In[61]:


final_data = pd.get_dummies(data)
print(final_data.shape)
print(data.shape)


# In[62]:


final_data.head()


# In[63]:


X = final_data.drop(['SalePrice'], axis = 'columns')
y = final_data['SalePrice']


# In[64]:


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[65]:



xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)


# In[66]:


xgboost.fit(X_train, y_train)
y_pred = xgboost.predict(X_test)
score = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('R2 Score: {:.4f}'.format(score))
print('RMSE: {:.4f}'.format(rmse))

# R2 Score: 0.9226
# RMSE: 0.1103


# In[ ]:


# K Fold Cross validation

scores = cross_val_score(xgboost, X, y, cv = 10)
scores


# In[ ]:


print('R2 Score: {:.4f}'.format(scores.mean()))

