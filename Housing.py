#!/usr/bin/env python
# coding: utf-8

# In[2]:


1、下载数据


# In[7]:


import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data=fetch_housing_data()


# In[8]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[ ]:


2、快速查看数据结构


# In[9]:


#加载数据
housing=load_housing_data()
#查看数据前5项
housing.head()


# In[10]:


#快速查看数据的描述，特别是总行数、每个属性的类型和非空值的数量
housing.info()


# In[11]:


#查看类别的属性
housing['ocean_proximity'].value_counts()


# In[12]:


#展示了数值属性的概括
housing.describe()


# In[13]:


#画出每个数值属性的柱状图
#Jupyter 的魔术命令%matplotlib inline。它会告诉 Jupyter 设定好 Matplotlib，以使用 Jupyter 自己的后端。绘图就会在 notebook 中渲染了
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[ ]:


3、创建测试集


# In[14]:


import hashlib
import numpy as np
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[15]:


housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[16]:


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[17]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[18]:


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[19]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[20]:


housing["income_cat"].value_counts() / len(housing)


# In[21]:


for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# In[ ]:


4、数据探索和可视化、发现规律


# In[22]:


#创建一个副本，以免损伤训练集
import copy
housing = strat_train_set.copy()


# In[23]:


#地理数据可视化
#创建一个所有街区的散点图
housing.plot(kind="scatter", x="longitude", y="latitude")


# In[24]:


#将alpha设为 0.1，可以更容易看出数据点的密度
housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)


# In[25]:


#房价数据展示
#每个圈的半径表示街区的人口（选项s），颜色代表价格（选项c）。
#我们用预先定义的名为jet的颜色图（选项cmap），它的范围是从蓝色（低价）到红色（高价）
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
             s=housing['population']/100,
             label='population',
             c='median_house_value',
             cmap=plt.get_cmap("jet"),
             colorbar=True,)
plt.legend()


# In[26]:


#查找关联
#使用corr()方法计算出每对属性间的标准相关系数
corr_matrix=housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# In[27]:


#使用 Pandas 的scatter_matrix函数检测属性间相关系数, 散点矩阵
import pandas
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[28]:


#预测房价中位数的属性是收入中位数，因此将这张图放大
housing.plot(kind="scatter", x="median_income",y="median_house_value",
             alpha=0.2)


# In[ ]:


5、属性组合试验


# In[29]:


#创建一些些新的属性
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix=housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# In[ ]:


6、为机器学习算法准备数据


# In[30]:


#
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[31]:


#数据清洗

#total_bedrooms有一些缺失值。有三个解决选项：

#去掉对应的街区；

#去掉整个属性；

#进行赋值（0、平均值、中位数等等）

#housing.dropna(subset=["total_bedrooms"])    # 选项1

#housing.drop("total_bedrooms", axis=1)       # 选项2

#median = housing["total_bedrooms"].median()
#housing["total_bedrooms"].fillna(median)     # 选项3 

#Scikit-Learn 提供了一个方便的类来处理缺失值：Imputer

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median") #指定用某属性的中位数来替换该属性所有的缺失值


# In[34]:


# 因为只有数值属性才能算出中位数，我们需要创建一份不包括文本属性ocean_proximity的数据副本
housing_num = housing.drop("ocean_proximity", axis=1)

#现在，就可以用fit()方法将imputer实例拟合到训练数据：

imputer.fit(housing_num)

#imputer计算出了每个属性的中位数，并将结果保存在了实例变量statistics_中。
#虽然此时只有属性total_bedrooms存在缺失值，但我们不能确定在以后的新的数据中会不会有其他属性也存在缺失值，
#所以安全的做法是将imputer应用到每个数值

imputer.statistics_


# In[35]:


housing_num.median().values


# In[37]:


#现在，你就可以使用这个“训练过的”imputer来对训练集进行转换，将缺失值替换为中位数：

X = imputer.transform(housing_num)

#结果是一个包含转换后特征的普通的 Numpy 数组。如果你想将其放回到 PandasDataFrame中，也很简单：

housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# In[38]:


#处理文本和类别属性


# In[ ]:




