# Ex-06-Feature-Transformation

# Aim:
1.To read and perform feature transformation for the given dataset.

# Explanation:
Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column (feature) and transform the values, which are useful for our further analysis. It is a technique by which we can boost our model performance.

# Algorithm:
### STEP 1
Read the given Data

### STEP 2
Clean the Data Set using Data Cleaning Process

### STEP 3
Apply Feature Transformation techniques to all the features of the data set

### STEP 4
Save the data to the file

# Program:
## Program developed by : V.NAVYA

## Register numnber : 212221230069


```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()


```
# Output:

## Dataset:

![image](https://user-images.githubusercontent.com/94165327/198091524-f65700b6-1059-4c3b-b676-cccb5457b197.png)

## Head:

![image](https://user-images.githubusercontent.com/94165327/198092009-d59e24db-e468-46b4-97a0-8011e18dca6a.png)

## Null data:

![image](https://user-images.githubusercontent.com/94165327/198092220-df332c09-a46d-49cd-b624-2571ae607242.png)

## Information:

![image](https://user-images.githubusercontent.com/94165327/198092483-82e7033d-2bd2-4347-878e-746deb2645f8.png)

## Description:

![image](https://user-images.githubusercontent.com/94165327/198092609-d59473ad-f146-4aa5-93bc-c1483600b8c7.png)

## Highly Positive Skew:

![image](https://user-images.githubusercontent.com/94165327/198092718-87fed493-046e-4f95-96a9-e6378b17a17b.png)

## Highly Negative Skew:

![image](https://user-images.githubusercontent.com/94165327/198092852-d4468155-70b6-49a4-91ce-b9a36f2994b5.png)

## Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/94165327/198092943-03b3ff74-75bc-41f0-b885-afc185b5b99f.png)

## Moderate Negative Skew:

![image](https://user-images.githubusercontent.com/94165327/198093051-a695ca7d-2ac0-430c-b403-567208120d23.png)

## Log of Highly Positive Skew:

![image](https://user-images.githubusercontent.com/94165327/198093158-75609289-ab54-4317-91ce-87b17f527349.png)

## Log of Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/94165327/198093260-bfcd3b66-b0b1-44c9-a810-41301db7c8b6.png)

## Reciprocal of Highly Positive Skew:

![image](https://user-images.githubusercontent.com/94165327/198093381-db20ae5c-2db6-414a-853a-ba6429e5abc9.png)


## Square root tranformation:

![image](https://user-images.githubusercontent.com/94165327/198093569-f770f47c-cd63-4e3d-9637-264110e6307d.png)


## Power transformation of Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/94165327/198093724-de65ae2e-c391-46aa-97e2-8eeba09802c9.png)

## Power transformation of Moderate Negative Skew:

![image](https://user-images.githubusercontent.com/94165327/198093845-9ef98b89-b69f-44e6-a2b4-5100a71f182f.png)

## Quantile transformation:

![image](https://user-images.githubusercontent.com/94165327/198093953-b134efc1-eb1e-433c-8a07-5ee5c45f067d.png)


#  Result:


Thus, Feature transformation is performed and executed successfully for the given dataset.












