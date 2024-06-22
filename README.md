Data science project to predict the price of vehicles taking into account their characteristics and using machine learning to design a model of prediction.

This project is made in Jupyther notebooks. Using python, Pandas, Numpy, Matplotlib, and scikit-learn.

```python
#importing data using pandas:

import pandas as pd

data = ("imports-85 (1).csv")

df = pd.read_csv(data)

#puting titles to each column:
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-weels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-rpm','price']

df.columns=headers

#see the top 5 rows:
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-weels</th>
      <th>engine-location</th>
      <th>weel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-rpm</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>?</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>15250</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python

headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-weels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-rpm','price']

df.columns=headers

df.dtypes #<--#see the data type of each column:
```




    symboling              int64
    normalized-losses     object
    make                  object
    fuel-type             object
    aspiration            object
    num-of-doors          object
    body-style            object
    drive-weels           object
    engine-location       object
    weel-base            float64
    length               float64
    width                float64
    height               float64
    curb-weight            int64
    engine-type           object
    num-of-cylinderss     object
    engine-size            int64
    fuel-system           object
    bore                  object
    stroke                object
    compression-ratio    float64
    horsepower            object
    peak-rpm              object
    city-mpg               int64
    highway-rpm            int64
    price                 object
    dtype: object




```python
#see the caracteristics of each numeric column:

df.columns=headers
df.describe() #<---return values that describes the dsta set (numeric columns only)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>weel-base</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>curb-weight</th>
      <th>engine-size</th>
      <th>compression-ratio</th>
      <th>city-mpg</th>
      <th>highway-rpm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.823529</td>
      <td>98.806373</td>
      <td>174.075000</td>
      <td>65.916667</td>
      <td>53.749020</td>
      <td>2555.602941</td>
      <td>126.892157</td>
      <td>10.148137</td>
      <td>25.240196</td>
      <td>30.769608</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.239035</td>
      <td>5.994144</td>
      <td>12.362123</td>
      <td>2.146716</td>
      <td>2.424901</td>
      <td>521.960820</td>
      <td>41.744569</td>
      <td>3.981000</td>
      <td>6.551513</td>
      <td>6.898337</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
      <td>86.600000</td>
      <td>141.100000</td>
      <td>60.300000</td>
      <td>47.800000</td>
      <td>1488.000000</td>
      <td>61.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>94.500000</td>
      <td>166.300000</td>
      <td>64.075000</td>
      <td>52.000000</td>
      <td>2145.000000</td>
      <td>97.000000</td>
      <td>8.575000</td>
      <td>19.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>97.000000</td>
      <td>173.200000</td>
      <td>65.500000</td>
      <td>54.100000</td>
      <td>2414.000000</td>
      <td>119.500000</td>
      <td>9.000000</td>
      <td>24.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>102.400000</td>
      <td>183.200000</td>
      <td>66.900000</td>
      <td>55.500000</td>
      <td>2939.250000</td>
      <td>142.000000</td>
      <td>9.400000</td>
      <td>30.000000</td>
      <td>34.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>120.900000</td>
      <td>208.100000</td>
      <td>72.300000</td>
      <td>59.800000</td>
      <td>4066.000000</td>
      <td>326.000000</td>
      <td>23.000000</td>
      <td>49.000000</td>
      <td>54.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#see the caracteristics of all columns:
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-weels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-rpm','price']

df.columns=headers
df.describe(include='all')#<---return values that describes the dsta set 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-weels</th>
      <th>engine-location</th>
      <th>weel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-rpm</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>204.000000</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204.000000</td>
      <td>...</td>
      <td>204.000000</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204.000000</td>
      <td>204</td>
      <td>204</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>52</td>
      <td>22</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>8</td>
      <td>39</td>
      <td>37</td>
      <td>NaN</td>
      <td>60</td>
      <td>24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>186</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>?</td>
      <td>toyota</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>mpfi</td>
      <td>3.62</td>
      <td>3.40</td>
      <td>NaN</td>
      <td>68</td>
      <td>5500</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>?</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>40</td>
      <td>32</td>
      <td>184</td>
      <td>167</td>
      <td>114</td>
      <td>96</td>
      <td>120</td>
      <td>201</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>93</td>
      <td>23</td>
      <td>20</td>
      <td>NaN</td>
      <td>19</td>
      <td>37</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.823529</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98.806373</td>
      <td>...</td>
      <td>126.892157</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.148137</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.240196</td>
      <td>30.769608</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.239035</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.994144</td>
      <td>...</td>
      <td>41.744569</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.981000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.551513</td>
      <td>6.898337</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>86.600000</td>
      <td>...</td>
      <td>61.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.000000</td>
      <td>16.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>94.500000</td>
      <td>...</td>
      <td>97.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.575000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.000000</td>
      <td>25.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>97.000000</td>
      <td>...</td>
      <td>119.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.000000</td>
      <td>30.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>102.400000</td>
      <td>...</td>
      <td>142.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.400000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.000000</td>
      <td>34.500000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>120.900000</td>
      <td>...</td>
      <td>326.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>49.000000</td>
      <td>54.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 26 columns</p>
</div>




```python
#check data type and overall info of the dataframe


df.columns=headers
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 204 entries, 0 to 203
    Data columns (total 26 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   symboling          204 non-null    int64  
     1   normalized-losses  204 non-null    object 
     2   make               204 non-null    object 
     3   fuel-type          204 non-null    object 
     4   aspiration         204 non-null    object 
     5   num-of-doors       204 non-null    object 
     6   body-style         204 non-null    object 
     7   drive-weels        204 non-null    object 
     8   engine-location    204 non-null    object 
     9   weel-base          204 non-null    float64
     10  length             204 non-null    float64
     11  width              204 non-null    float64
     12  height             204 non-null    float64
     13  curb-weight        204 non-null    int64  
     14  engine-type        204 non-null    object 
     15  num-of-cylinderss  204 non-null    object 
     16  engine-size        204 non-null    int64  
     17  fuel-system        204 non-null    object 
     18  bore               204 non-null    object 
     19  stroke             204 non-null    object 
     20  compression-ratio  204 non-null    float64
     21  horsepower         204 non-null    object 
     22  peak-rpm           204 non-null    object 
     23  city-mpg           204 non-null    int64  
     24  highway-rpm        204 non-null    int64  
     25  price              204 non-null    object 
    dtypes: float64(5), int64(5), object(16)
    memory usage: 41.6+ KB
    


```python
#acces and add to a column values]

df.columns=headers
df['symboling']=df['symboling']+1 #<--added 1 to every symboling values
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-weels</th>
      <th>engine-location</th>
      <th>weel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-rpm</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>?</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>15250</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>199</th>
      <td>0</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>23</td>
      <td>28</td>
      <td>16845</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>8.7</td>
      <td>160</td>
      <td>5300</td>
      <td>19</td>
      <td>25</td>
      <td>19045</td>
    </tr>
    <tr>
      <th>201</th>
      <td>0</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>173</td>
      <td>mpfi</td>
      <td>3.58</td>
      <td>2.87</td>
      <td>8.8</td>
      <td>134</td>
      <td>5500</td>
      <td>18</td>
      <td>23</td>
      <td>21485</td>
    </tr>
    <tr>
      <th>202</th>
      <td>0</td>
      <td>95</td>
      <td>volvo</td>
      <td>diesel</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>145</td>
      <td>idi</td>
      <td>3.01</td>
      <td>3.40</td>
      <td>23.0</td>
      <td>106</td>
      <td>4800</td>
      <td>26</td>
      <td>27</td>
      <td>22470</td>
    </tr>
    <tr>
      <th>203</th>
      <td>0</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>19</td>
      <td>25</td>
      <td>22625</td>
    </tr>
  </tbody>
</table>
<p>204 rows × 26 columns</p>
</div>




```python
#Dropping Missing Values

df.columns=headers
df.dropna(subset=["price"], axis=0, inplace=True) #<--Delete the rows that contains missing (N/A) values in column 'price'
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-weels</th>
      <th>engine-location</th>
      <th>weel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-rpm</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>204.000000</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204.000000</td>
      <td>...</td>
      <td>204.000000</td>
      <td>204</td>
      <td>204</td>
      <td>204</td>
      <td>204.000000</td>
      <td>204</td>
      <td>204</td>
      <td>204.000000</td>
      <td>204.000000</td>
      <td>204</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>52</td>
      <td>22</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>8</td>
      <td>39</td>
      <td>37</td>
      <td>NaN</td>
      <td>60</td>
      <td>24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>186</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>?</td>
      <td>toyota</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>mpfi</td>
      <td>3.62</td>
      <td>3.40</td>
      <td>NaN</td>
      <td>68</td>
      <td>5500</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>?</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>40</td>
      <td>32</td>
      <td>184</td>
      <td>167</td>
      <td>114</td>
      <td>96</td>
      <td>120</td>
      <td>201</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>93</td>
      <td>23</td>
      <td>20</td>
      <td>NaN</td>
      <td>19</td>
      <td>37</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.823529</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>98.806373</td>
      <td>...</td>
      <td>126.892157</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.148137</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25.240196</td>
      <td>30.769608</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.239035</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.994144</td>
      <td>...</td>
      <td>41.744569</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.981000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.551513</td>
      <td>6.898337</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>86.600000</td>
      <td>...</td>
      <td>61.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.000000</td>
      <td>16.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>94.500000</td>
      <td>...</td>
      <td>97.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.575000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.000000</td>
      <td>25.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>97.000000</td>
      <td>...</td>
      <td>119.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.000000</td>
      <td>30.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>102.400000</td>
      <td>...</td>
      <td>142.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.400000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.000000</td>
      <td>34.500000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>120.900000</td>
      <td>...</td>
      <td>326.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>49.000000</td>
      <td>54.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 26 columns</p>
</div>




```python
#Dropping Missing Values

df.columns=headers
df.dropna(subset=[1], axis=1, inplace=True) #<--Delete the column that contains missing (N/A) values in row 1
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-weels</th>
      <th>engine-location</th>
      <th>weel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-rpm</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>?</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>15250</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>199</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>23</td>
      <td>28</td>
      <td>16845</td>
    </tr>
    <tr>
      <th>200</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>8.7</td>
      <td>160</td>
      <td>5300</td>
      <td>19</td>
      <td>25</td>
      <td>19045</td>
    </tr>
    <tr>
      <th>201</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>173</td>
      <td>mpfi</td>
      <td>3.58</td>
      <td>2.87</td>
      <td>8.8</td>
      <td>134</td>
      <td>5500</td>
      <td>18</td>
      <td>23</td>
      <td>21485</td>
    </tr>
    <tr>
      <th>202</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>diesel</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>145</td>
      <td>idi</td>
      <td>3.01</td>
      <td>3.40</td>
      <td>23.0</td>
      <td>106</td>
      <td>4800</td>
      <td>26</td>
      <td>27</td>
      <td>22470</td>
    </tr>
    <tr>
      <th>203</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>19</td>
      <td>25</td>
      <td>22625</td>
    </tr>
  </tbody>
</table>
<p>204 rows × 26 columns</p>
</div>




```python
#Dropping an spesific Missing Values

df.columns=headers
df.drop(df.index[df['compression-ratio'] == 9.0], inplace=True)#<-- drop the rows where compression-ratio is 9.0
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-weels</th>
      <th>engine-location</th>
      <th>weel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-rpm</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>?</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>15250</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>158</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>105.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>17710</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>?</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>wagon</td>
      <td>fwd</td>
      <td>front</td>
      <td>105.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>18920</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>199</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>23</td>
      <td>28</td>
      <td>16845</td>
    </tr>
    <tr>
      <th>200</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>8.7</td>
      <td>160</td>
      <td>5300</td>
      <td>19</td>
      <td>25</td>
      <td>19045</td>
    </tr>
    <tr>
      <th>201</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>173</td>
      <td>mpfi</td>
      <td>3.58</td>
      <td>2.87</td>
      <td>8.8</td>
      <td>134</td>
      <td>5500</td>
      <td>18</td>
      <td>23</td>
      <td>21485</td>
    </tr>
    <tr>
      <th>202</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>diesel</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>145</td>
      <td>idi</td>
      <td>3.01</td>
      <td>3.40</td>
      <td>23.0</td>
      <td>106</td>
      <td>4800</td>
      <td>26</td>
      <td>27</td>
      <td>22470</td>
    </tr>
    <tr>
      <th>203</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>19</td>
      <td>25</td>
      <td>22625</td>
    </tr>
  </tbody>
</table>
<p>159 rows × 26 columns</p>
</div>




```python
#Replasing missing values (by the mean value of the column)
import numpy as np

df.columns=headers

missingvalueformat = '?'                                            #<----Find the form of the missing values on the data

df["normalized-losses"].replace(missingvalueformat, 0, inplace=True)#<----Subtitute the missing value by a numeric value
df['normalized-losses']=df['normalized-losses'].astype('int')       #<--- Change the format of the column/feature to 'int'
meanvalue = df['normalized-losses'].mean()                          #<---- Find the mean() of the column/feature 
df["normalized-losses"].replace(0, meanvalue, inplace=True)         #<---- Replace the values 0 by the mean value
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-weels</th>
      <th>engine-location</th>
      <th>weel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-rpm</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>98.078431</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>21</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>98.078431</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>19</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>164.000000</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164.000000</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>22</td>
      <td>17450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>98.078431</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>25</td>
      <td>15250</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>199</th>
      <td>-1</td>
      <td>95.000000</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>23</td>
      <td>28</td>
      <td>16845</td>
    </tr>
    <tr>
      <th>200</th>
      <td>-1</td>
      <td>95.000000</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>8.7</td>
      <td>160</td>
      <td>5300</td>
      <td>19</td>
      <td>25</td>
      <td>19045</td>
    </tr>
    <tr>
      <th>201</th>
      <td>-1</td>
      <td>95.000000</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>173</td>
      <td>mpfi</td>
      <td>3.58</td>
      <td>2.87</td>
      <td>8.8</td>
      <td>134</td>
      <td>5500</td>
      <td>18</td>
      <td>23</td>
      <td>21485</td>
    </tr>
    <tr>
      <th>202</th>
      <td>-1</td>
      <td>95.000000</td>
      <td>volvo</td>
      <td>diesel</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>145</td>
      <td>idi</td>
      <td>3.01</td>
      <td>3.40</td>
      <td>23.0</td>
      <td>106</td>
      <td>4800</td>
      <td>26</td>
      <td>27</td>
      <td>22470</td>
    </tr>
    <tr>
      <th>203</th>
      <td>-1</td>
      <td>95.000000</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>19</td>
      <td>25</td>
      <td>22625</td>
    </tr>
  </tbody>
</table>
<p>204 rows × 26 columns</p>
</div>




```python
#Apply calculations to all values in a column

df.columns=headers
df["city-mpg"]= 235/df["city-mpg"] #<---change the column values to kilometers
df.rename(columns={"city-mpg":"city-L/100Km"}, inplace =True)#<---rename a column
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>fuel-type</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-weels</th>
      <th>engine-location</th>
      <th>weel-base</th>
      <th>...</th>
      <th>engine-size</th>
      <th>fuel-system</th>
      <th>bore</th>
      <th>stroke</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-L/100Km</th>
      <th>highway-rpm</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>130</td>
      <td>mpfi</td>
      <td>3.47</td>
      <td>2.68</td>
      <td>9.0</td>
      <td>111</td>
      <td>5000</td>
      <td>11.190476</td>
      <td>27</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>?</td>
      <td>alfa-romero</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>152</td>
      <td>mpfi</td>
      <td>2.68</td>
      <td>3.47</td>
      <td>9.0</td>
      <td>154</td>
      <td>5000</td>
      <td>12.368421</td>
      <td>26</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>109</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>10.0</td>
      <td>102</td>
      <td>5500</td>
      <td>9.791667</td>
      <td>30</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.0</td>
      <td>115</td>
      <td>5500</td>
      <td>13.055556</td>
      <td>22</td>
      <td>17450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>?</td>
      <td>audi</td>
      <td>gas</td>
      <td>std</td>
      <td>two</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>136</td>
      <td>mpfi</td>
      <td>3.19</td>
      <td>3.40</td>
      <td>8.5</td>
      <td>110</td>
      <td>5500</td>
      <td>12.368421</td>
      <td>25</td>
      <td>15250</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>199</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>10.217391</td>
      <td>28</td>
      <td>16845</td>
    </tr>
    <tr>
      <th>200</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>8.7</td>
      <td>160</td>
      <td>5300</td>
      <td>12.368421</td>
      <td>25</td>
      <td>19045</td>
    </tr>
    <tr>
      <th>201</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>173</td>
      <td>mpfi</td>
      <td>3.58</td>
      <td>2.87</td>
      <td>8.8</td>
      <td>134</td>
      <td>5500</td>
      <td>13.055556</td>
      <td>23</td>
      <td>21485</td>
    </tr>
    <tr>
      <th>202</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>diesel</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>145</td>
      <td>idi</td>
      <td>3.01</td>
      <td>3.40</td>
      <td>23.0</td>
      <td>106</td>
      <td>4800</td>
      <td>9.038462</td>
      <td>27</td>
      <td>22470</td>
    </tr>
    <tr>
      <th>203</th>
      <td>-1</td>
      <td>95</td>
      <td>volvo</td>
      <td>gas</td>
      <td>turbo</td>
      <td>four</td>
      <td>sedan</td>
      <td>rwd</td>
      <td>front</td>
      <td>109.1</td>
      <td>...</td>
      <td>141</td>
      <td>mpfi</td>
      <td>3.78</td>
      <td>3.15</td>
      <td>9.5</td>
      <td>114</td>
      <td>5400</td>
      <td>12.368421</td>
      <td>25</td>
      <td>22625</td>
    </tr>
  </tbody>
</table>
<p>204 rows × 26 columns</p>
</div>




```python
#Change the format of column elements

df.columns=headers
missingvalueformat = '?'
# df["price"].replace(np.nan, 0, inplace=True)
df["price"].replace(missingvalueformat, 0, inplace=True)
df["price"]=df["price"].astype("int")
df["price"]

```




    0      16500
    1      16500
    2      13950
    3      17450
    4      15250
           ...  
    199    16845
    200    19045
    201    21485
    202    22470
    203    22625
    Name: price, Length: 204, dtype: int32




```python
#Data normalization: Convert big values to a closer range to compare with other data.


#Simple Scalling Method: Xnew = Xold/Xmax {0 to 1}

import pandas as pd
import numpy as np

data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-weels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-rpm','price']

df.columns=headers
df['length']=df['length']/df['length'].max()#<--Xnew = Xold/Xmax
df['length']
```




    0      0.811148
    1      0.822681
    2      0.848630
    3      0.848630
    4      0.851994
             ...   
    199    0.907256
    200    0.907256
    201    0.907256
    202    0.907256
    203    0.907256
    Name: length, Length: 204, dtype: float64




```python
#Min-Max: Xnew = (Xold-Xmin)/(Xmax-Xmin) {0 to 1}

df.columns=headers
df['length']=(df['length']-df['length'].min())/(df['length'].max()-df['length'].min())#<--Xnew = (Xold-Xmin)/(Xmax-Xmin)
df['length']
```




    0      0.413433
    1      0.449254
    2      0.529851
    3      0.529851
    4      0.540299
             ...   
    199    0.711940
    200    0.711940
    201    0.711940
    202    0.711940
    203    0.711940
    Name: length, Length: 204, dtype: float64




```python
#Z-Score: Xnew = (Xold-µ)/σ ----(  µ = df[X].mean()   σ = df[X].std()  ) {-3 to 3}

import pandas as pd
import numpy as np

data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-weels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-rpm','price']

df.columns=headers
df['length']=(df['length']-df['length'].mean())/df['length'].std()#<--Xnew = (Xold-µ)/σ
df['length']
```




    0     -0.426707
    1     -0.232565
    2      0.204253
    3      0.204253
    4      0.260878
             ...   
    199    1.191138
    200    1.191138
    201    1.191138
    202    1.191138
    203    1.191138
    Name: length, Length: 204, dtype: float64




```python
#Binning and adding a column

df.columns=headers

#Change the format of column elements:

missingvalueformat = '?'
df["price"].replace(missingvalueformat, 0, inplace=True)
df["price"]=df["price"].astype("int")
df["price"]

#Binning and adding a column

binn = np.linspace(min(df["price"]), max(df["price"]),4) #<--- intervals: numpy.linspace ((min value), (max value), number of groups (intervals) +1)
pricebinnames = ['Low', 'Mid', 'High'] #<------name of the groups (in assendant order)
df['price_binned'] = pd.cut(df['price'], binn, labels=pricebinnames, include_lowest = True)#<-- add a new column pd.cut(array to be binned, intervals, groups names, include_lowest value)
# df['price'],df['price_binned']
df['price_binned']

```




    0      Mid
    1      Mid
    2      Low
    3      Mid
    4      Mid
          ... 
    199    Mid
    200    Mid
    201    Mid
    202    Mid
    203    Mid
    Name: price_binned, Length: 204, dtype: category
    Categories (3, object): ['Low' < 'Mid' < 'High']




```python
#Turn categorical or object values into cuantizable data (with no change any value format)

df.columns=headers
pd.get_dummies(df['fuel-type']) 

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diesel</th>
      <th>gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>199</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>201</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>202</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>203</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>204 rows × 2 columns</p>
</div>




```python
# Count Elements on categorical variable 

df.columns=headers
# create a variable with the counted values:
count_variable = df['drive-wheels'].value_counts()
print(count_variable)


```

    fwd    120
    rwd     75
    4wd      9
    Name: drive-wheels, dtype: int64
    


```python
# Graph 
#Boxplot with seaborn

import pandas as pd
import matplotlib as mpl
import seaborn as sns
import numpy as np

data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-rpm','price']

df.columns=headers

#Change the format of column elements: seaborn only works if at least one of the elements is on int64 or float64 type

df["price"].replace(missingvalueformat, 0, inplace=True)
df["price"]=df["price"].astype("int")

sns.boxplot(x='drive-wheels', y='price', data=df)
```




    <AxesSubplot:xlabel='drive-wheels', ylabel='price'>




    
![png](output_18_1.png)
    



```python
# Graph
#Scatter Plot with Matplolib


import matplotlib.pyplot as plt
import seaborn as sns

df.columns=headers

#Change the format of column elements: Matplotlib only works if at least one of the elements is on int64 or float64 type

df["price"].replace(missingvalueformat, 0, inplace=True)
df["price"]=df["price"].astype("int")

y=df['engine-size'] #<---target variable on the y-axis
x=df['price']#<-----predictor variable on the x-axis
plt.scatter(x, y)
plt.title('Scatter Grapg Relationship between engine size and price')
plt.xlabel('Price')
plt.ylabel('Engine Size')

#as seen on the graph is notisable how is assendant the relationship between price and engine size, so the biggesr the engine the more the price.


```




    Text(0, 0.5, 'Engine Size')




    
![png](output_19_1.png)
    



```python
# Comparing by Grouping with groupby()

df.columns=headers
df["price"].replace("?", 0, inplace=True)
df["price"]=df["price"].astype("int")

#finding the average price of vehicles and observe how they differ between different types of “body-style” and “drive-wheels”variables.

df_test = df[["drive-wheels", 'body-style', 'price']] #<--pick up the columns to compare
df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()#<- take the mean of the compared "price" to see how the average price differs across the board
df_grp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>drive-wheels</th>
      <th>body-style</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4wd</td>
      <td>hatchback</td>
      <td>3801.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4wd</td>
      <td>sedan</td>
      <td>12647.333333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4wd</td>
      <td>wagon</td>
      <td>9095.750000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fwd</td>
      <td>convertible</td>
      <td>11595.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fwd</td>
      <td>hardtop</td>
      <td>8249.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>fwd</td>
      <td>hatchback</td>
      <td>8396.387755</td>
    </tr>
    <tr>
      <th>6</th>
      <td>fwd</td>
      <td>sedan</td>
      <td>9467.526316</td>
    </tr>
    <tr>
      <th>7</th>
      <td>fwd</td>
      <td>wagon</td>
      <td>9997.333333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>rwd</td>
      <td>convertible</td>
      <td>26563.250000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>rwd</td>
      <td>hardtop</td>
      <td>24202.714286</td>
    </tr>
    <tr>
      <th>10</th>
      <td>rwd</td>
      <td>hatchback</td>
      <td>13583.157895</td>
    </tr>
    <tr>
      <th>11</th>
      <td>rwd</td>
      <td>sedan</td>
      <td>21711.833333</td>
    </tr>
    <tr>
      <th>12</th>
      <td>rwd</td>
      <td>wagon</td>
      <td>16994.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualizing table with pivot()

df.columns=headers
df["price"].replace("?", 0, inplace=True)
df["price"]=df["price"].astype("int")

#finding the average price of vehicles and observe how they differ between different types of “body-style” and “drive-wheels”variables.

df_test = df[["drive-wheels", 'body-style', 'price']] #<--pick up the columns to compare
df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()#<- take the mean of the compared "price" to see how the average price differs across the board
df_pivot = df_grp.pivot(index="drive-wheels", columns='body-style')
df_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">price</th>
    </tr>
    <tr>
      <th>body-style</th>
      <th>convertible</th>
      <th>hardtop</th>
      <th>hatchback</th>
      <th>sedan</th>
      <th>wagon</th>
    </tr>
    <tr>
      <th>drive-wheels</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4wd</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>3801.500000</td>
      <td>12647.333333</td>
      <td>9095.750000</td>
    </tr>
    <tr>
      <th>fwd</th>
      <td>11595.00</td>
      <td>8249.000000</td>
      <td>8396.387755</td>
      <td>9467.526316</td>
      <td>9997.333333</td>
    </tr>
    <tr>
      <th>rwd</th>
      <td>26563.25</td>
      <td>24202.714286</td>
      <td>13583.157895</td>
      <td>21711.833333</td>
      <td>16994.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualizing Heatmap() graph

df.columns=headers
df["price"].replace("?", 0, inplace=True)
df["price"]=df["price"].astype("int")

#finding the average price of vehicles and observe how they differ between different types of “body-style” and “drive-wheels”variables.

df_test = df[["drive-wheels", 'body-style', 'price']] #<--pick up the columns to compare
df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()#<- take the mean of the compared "price" to see how the average price differs across the board
df_pivot = df_grp.pivot(index="drive-wheels", columns='body-style')

plt.pcolor(df_pivot,cmap='RdBu')
plt.colorbar()
plt.show()
df_pivot
```


    
![png](output_22_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">price</th>
    </tr>
    <tr>
      <th>body-style</th>
      <th>convertible</th>
      <th>hardtop</th>
      <th>hatchback</th>
      <th>sedan</th>
      <th>wagon</th>
    </tr>
    <tr>
      <th>drive-wheels</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4wd</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>3801.500000</td>
      <td>12647.333333</td>
      <td>9095.750000</td>
    </tr>
    <tr>
      <th>fwd</th>
      <td>11595.00</td>
      <td>8249.000000</td>
      <td>8396.387755</td>
      <td>9467.526316</td>
      <td>9997.333333</td>
    </tr>
    <tr>
      <th>rwd</th>
      <td>26563.25</td>
      <td>24202.714286</td>
      <td>13583.157895</td>
      <td>21711.833333</td>
      <td>16994.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ANOVA (Analysis Of VAriance) between a categorical variable and a numerical variable

import scipy
from scipy import *
import matplotlib.pyplot as plt


df.columns=headers
# convert price to numrrical values:
df["price"].replace("?", 0, inplace=True)
df["price"]=df["price"].astype("int")

# Apply ANOVA: Using scipy methods:

df_anova=df[["make", "price"]]#<------pick up the columns ["categorical", "numeric"]
grouped_anova=df_anova.groupby(["make"])#<----Group the categorical variable's categories

#------------between "honda" and "subaru" prices
anova_results_l=stats.f_oneway(grouped_anova.get_group("honda")["price"],grouped_anova.get_group("subaru")["price"])#<--perform the ANOVA using stats.f_oneway(categorical_groups.get_group("category1")["numerical_feature"],categorical_groups.get_group("category2")["numerical_feature"])
anova_results_l

#Result Description: 
#statistic = F-test Score: ratio of variance between groups means and each categorie. The less the result teh less varianve is between the 2 categories ("honda" and "subaru")
#pvalue:shows if the result represent a significant difference on the analysis. the larger the more signifficant is the result

#------------between "honda" and "jaguar" price
anova_results_j=stats.f_oneway(grouped_anova.get_group("honda")["price"],grouped_anova.get_group("jaguar")["price"])#<--perform the ANOVA using stats.f_oneway(categorical_groups.get_group("category1")["numerical_feature"],categorical_groups.get_group("category2")["numerical_feature"])
anova_results_j

```




    F_onewayResult(statistic=400.925870564337, pvalue=1.0586193512077862e-11)




```python
# Correlation between two variables/columns/features

import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-rpm','price']

df.columns=headers

# convert non-numerical to numrrical values:
df["price"].replace("?", 0, inplace=True)
df["price"]=df["price"].astype("int")

df['peak-rpm'].replace("?", 0, inplace=True)
df['peak-rpm']=df['peak-rpm'].astype("float")



#Create Graph 1
# sns.regplot(x=df['engine-size'],y=df["price"] ) #<--Point graph with seaborn.regplot
# plt.ylim(0,)#<----Draw a medial line within the dots
#Graph 1 description: Ascendant correlation, so the more engine-size the more the price(positive correlation)


#Create Graph 2
# sns.regplot(x=df["highway-rpm"],y=df['price'] )
# plt.ylim(0,)
#Graph 2 description: Descendant correlation, so the more highway-rpm the less the price (Negative correlation)


#Create Graph 3
# Eliminate the ceros from 'peak-rpm' and replace it by its mean vakue (so the graph will no have big empty spaces)
meanvalue = df['peak-rpm'].mean()
df['peak-rpm'].replace(0, meanvalue, inplace=True)

sns.regplot(x=df['peak-rpm'],y=df['price'] )
plt.ylim(0,)
#Graph 3 description: Weak correlation, not big change, so no influence between variables
```




    (0.0, 47670.0)




    
![png](output_24_1.png)
    



```python
#Pearsin Correlation: How correlated are two features

import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-rpm','price']

df.columns=headers

# convert non-numerical to numrrical values:

df["price"].replace("?", 0, inplace=True)#<----Subtitute the missing value by a numeric value
df["price"]=df["price"].astype("int")      #<--- Change the format of the column/feature to 'int'
meanvalue = df["price"].mean()                          #<---- Find the mean() of the column/feature 
df["price"].replace(0, meanvalue, inplace=True)

df['horsepower'].replace("?", 0, inplace=True)#<----Subtitute the missing value by a numeric value
df['horsepower']=df['horsepower'].astype("int")     #<--- Change the format of the column/feature to 'int'
meanvalue = df['horsepower'].mean()                          #<---- Find the mean() of the column/feature 
df['horsepower'].replace(0, meanvalue, inplace=True)







pearson_coef,p_value = stats.pearsonr(df['horsepower'], df["price"])
pearson_coef,p_value

#the result shows: 
#Pearson coefficient= 0.757... (closer to +1 = Large Positive correlation)
#p-value = 3.03...e-30 (<0.001 = strong certainly)
```




    (0.7573503297389315, 3.0304330693139256e-39)




```python
#Average X value of Y's categorical groups
import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-rpm','price']

df.columns=headers
df["price"].replace("?", 0, inplace=True)#<----Subtitute the missing value by a numeric value
df["price"]=df["price"].astype("int")      #<--- Change the format of the column/feature to 'int'
meanvalue = df["price"].mean()                          #<---- Find the mean() of the column/feature 
df["price"].replace(0, meanvalue, inplace=True)

# See average price for body style groups and rename a column price:
df.rename(columns={'price':'average_price'}, inplace =True)#<--Rename the column to represent the new values

df_test = df[['body-style','average_price']] #<--Pick the features to work with and store it on a variable
df_grp = df_test.groupby(['body-style'], as_index=False).mean()#<--

df_grp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>body-style</th>
      <th>average_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>convertible</td>
      <td>23569.600000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hardtop</td>
      <td>22208.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hatchback</td>
      <td>10042.850140</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sedan</td>
      <td>14428.234477</td>
    </tr>
    <tr>
      <th>4</th>
      <td>wagon</td>
      <td>12371.960000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Simple Linear Model Estimator (SLM)
import pandas as pd
# import seaborn as sns
# import scipy
# from scipy import *
# import matplotlib.pyplot as plt
from sklearn.linear_model import * #<--Import the method


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

# convert non-numerical to numrrical values:
df["price"].replace("?", 0, inplace=True)
df["price"]=df["price"].astype("int")    
meanvalue = df["price"].mean()                        
df["price"].replace(0, meanvalue, inplace=True)
# df["price"]=df["price"].astype("int") 

#create a LinearRegression() object:
lm = LinearRegression()

#Define variables:
X = df['highway-mpg']
y = df['price']

X = X.values.reshape(-1, 1) #<-- reshape the arrays to make it 2D instead of 1D
y = y.values.reshape(-1, 1)

#fit the model/find b0 and b1
lm.fit(X,y)

#See the predictor's prediction cahnged values:
yhat = lm.predict(X)

#calculate intercept (b0)
b0=lm.intercept_

#calculate slope (b1)
b1 = lm.coef_

#Visualize Results
print('predictions: ',yhat,'b0 (interceptor): ', b0, 'b1 (slope): ',b1)

#formula to predict a target value ('price') using one predictor (highway-mpg) as a trainer to fit:
#----------y=b0-b1*X---(price predicted = lm.intercept_-lm.coef_*'highway-mpg')
y2=b0-b1*X
y2

```

    predictions:  [[16180.17703462]
     [16970.59458398]
     [13808.92438655]
     [20132.26478141]
     [17761.01213334]
     [17761.01213334]
     [17761.01213334]
     [21713.09988012]
     [20132.26478141]
     [14599.34193591]
     [14599.34193591]
     [15389.75948527]
     [15389.75948527]
     [17761.01213334]
     [20132.26478141]
     [20132.26478141]
     [21713.09988012]
     [-4370.67924865]
     [ 3533.49624492]
     [ 3533.49624492]
     [ 5114.33134363]
     [ 7485.5839917 ]
     [13808.92438655]
     [ 7485.5839917 ]
     [ 7485.5839917 ]
     [ 7485.5839917 ]
     [13808.92438655]
     [13808.92438655]
     [18551.42968269]
     [-5161.09679801]
     [ 7485.5839917 ]
     [ 4323.91379427]
     [10647.25418913]
     [10647.25418913]
     [10647.25418913]
     [10647.25418913]
     [11437.67173848]
     [11437.67173848]
     [11437.67173848]
     [11437.67173848]
     [15389.75948527]
     [13018.5068372 ]
     [14599.34193591]
     [ 3533.49624492]
     [ 3533.49624492]
     [14599.34193591]
     [22503.51742948]
     [22503.51742948]
     [24084.35252819]
     [13018.5068372 ]
     [ 7485.5839917 ]
     [ 7485.5839917 ]
     [ 7485.5839917 ]
     [ 7485.5839917 ]
     [19341.84723205]
     [19341.84723205]
     [19341.84723205]
     [19341.84723205]
     [12228.08928784]
     [12228.08928784]
     [12228.08928784]
     [12228.08928784]
     [ 4323.91379427]
     [12228.08928784]
     [16180.17703462]
     [ 6695.16644234]
     [17761.01213334]
     [17761.01213334]
     [17761.01213334]
     [17761.01213334]
     [23293.93497883]
     [23293.93497883]
     [24874.77007755]
     [24874.77007755]
     [18551.42968269]
     [ 5114.33134363]
     [ 7485.5839917 ]
     [ 7485.5839917 ]
     [13808.92438655]
     [13808.92438655]
     [12228.08928784]
     [18551.42968269]
     [18551.42968269]
     [18551.42968269]
     [12228.08928784]
     [12228.08928784]
     [13808.92438655]
     [13808.92438655]
     [ 8276.00154106]
     [-1999.42660058]
     [ 8276.00154106]
     [ 8276.00154106]
     [ 8276.00154106]
     [ 8276.00154106]
     [ 8276.00154106]
     [ 8276.00154106]
     [ 8276.00154106]
     [ 8276.00154106]
     [10647.25418913]
     [10647.25418913]
     [20132.26478141]
     [20132.26478141]
     [17761.01213334]
     [17761.01213334]
     [19341.84723205]
     [17761.01213334]
     [18551.42968269]
     [11437.67173848]
     [18551.42968269]
     [17761.01213334]
     [18551.42968269]
     [11437.67173848]
     [18551.42968269]
     [17761.01213334]
     [18551.42968269]
     [11437.67173848]
     [18551.42968269]
     [ 5114.33134363]
     [13808.92438655]
     [ 7485.5839917 ]
     [ 7485.5839917 ]
     [ 7485.5839917 ]
     [13808.92438655]
     [18551.42968269]
     [16180.17703462]
     [17761.01213334]
     [17761.01213334]
     [17761.01213334]
     [15389.75948527]
     [13018.5068372 ]
     [13018.5068372 ]
     [15389.75948527]
     [15389.75948527]
     [15389.75948527]
     [15389.75948527]
     [16970.59458398]
     [16970.59458398]
     [ 9066.41909041]
     [13018.5068372 ]
     [13018.5068372 ]
     [ 8276.00154106]
     [11437.67173848]
     [12228.08928784]
     [17761.01213334]
     [14599.34193591]
     [12228.08928784]
     [13018.5068372 ]
     [14599.34193591]
     [19341.84723205]
     [ 6695.16644234]
     [ 7485.5839917 ]
     [ 7485.5839917 ]
     [ 8276.00154106]
     [12228.08928784]
     [12228.08928784]
     [ 8276.00154106]
     [ 8276.00154106]
     [ 9066.41909041]
     [  371.82604749]
     [  371.82604749]
     [10647.25418913]
     [10647.25418913]
     [10647.25418913]
     [10647.25418913]
     [14599.34193591]
     [14599.34193591]
     [13808.92438655]
     [13808.92438655]
     [13808.92438655]
     [13808.92438655]
     [13808.92438655]
     [13808.92438655]
     [10647.25418913]
     [11437.67173848]
     [12228.08928784]
     [12228.08928784]
     [12228.08928784]
     [18551.42968269]
     [18551.42968269]
     [18551.42968269]
     [18551.42968269]
     [ 1162.24359685]
     [10647.25418913]
     [ 1162.24359685]
     [10647.25418913]
     [10647.25418913]
     [ 4323.91379427]
     [12228.08928784]
     [14599.34193591]
     [14599.34193591]
     [18551.42968269]
     [ 7485.5839917 ]
     [13018.5068372 ]
     [15389.75948527]
     [15389.75948527]
     [15389.75948527]
     [15389.75948527]
     [20132.26478141]
     [20132.26478141]
     [15389.75948527]
     [17761.01213334]
     [19341.84723205]
     [16180.17703462]
     [17761.01213334]] b0 (interceptor):  [37521.45086725] b1 (slope):  [[-790.41754936]]
    




    array([[58862.72469988],
           [58072.30715053],
           [61233.97734795],
           [54910.6369531 ],
           [57281.88960117],
           [57281.88960117],
           [57281.88960117],
           [53329.80185439],
           [54910.6369531 ],
           [60443.5597986 ],
           [60443.5597986 ],
           [59653.14224924],
           [59653.14224924],
           [57281.88960117],
           [54910.6369531 ],
           [54910.6369531 ],
           [53329.80185439],
           [79413.58098316],
           [71509.40548959],
           [71509.40548959],
           [69928.57039088],
           [67557.31774281],
           [61233.97734795],
           [67557.31774281],
           [67557.31774281],
           [67557.31774281],
           [61233.97734795],
           [61233.97734795],
           [56491.47205181],
           [80203.99853251],
           [67557.31774281],
           [70718.98794023],
           [64395.64754538],
           [64395.64754538],
           [64395.64754538],
           [64395.64754538],
           [63605.22999602],
           [63605.22999602],
           [63605.22999602],
           [63605.22999602],
           [59653.14224924],
           [62024.39489731],
           [60443.5597986 ],
           [71509.40548959],
           [71509.40548959],
           [60443.5597986 ],
           [52539.38430503],
           [52539.38430503],
           [50958.54920632],
           [62024.39489731],
           [67557.31774281],
           [67557.31774281],
           [67557.31774281],
           [67557.31774281],
           [55701.05450246],
           [55701.05450246],
           [55701.05450246],
           [55701.05450246],
           [62814.81244667],
           [62814.81244667],
           [62814.81244667],
           [62814.81244667],
           [70718.98794023],
           [62814.81244667],
           [58862.72469988],
           [68347.73529216],
           [57281.88960117],
           [57281.88960117],
           [57281.88960117],
           [57281.88960117],
           [51748.96675567],
           [51748.96675567],
           [50168.13165696],
           [50168.13165696],
           [56491.47205181],
           [69928.57039088],
           [67557.31774281],
           [67557.31774281],
           [61233.97734795],
           [61233.97734795],
           [62814.81244667],
           [56491.47205181],
           [56491.47205181],
           [56491.47205181],
           [62814.81244667],
           [62814.81244667],
           [61233.97734795],
           [61233.97734795],
           [66766.90019345],
           [77042.32833509],
           [66766.90019345],
           [66766.90019345],
           [66766.90019345],
           [66766.90019345],
           [66766.90019345],
           [66766.90019345],
           [66766.90019345],
           [66766.90019345],
           [64395.64754538],
           [64395.64754538],
           [54910.6369531 ],
           [54910.6369531 ],
           [57281.88960117],
           [57281.88960117],
           [55701.05450246],
           [57281.88960117],
           [56491.47205181],
           [63605.22999602],
           [56491.47205181],
           [57281.88960117],
           [56491.47205181],
           [63605.22999602],
           [56491.47205181],
           [57281.88960117],
           [56491.47205181],
           [63605.22999602],
           [56491.47205181],
           [69928.57039088],
           [61233.97734795],
           [67557.31774281],
           [67557.31774281],
           [67557.31774281],
           [61233.97734795],
           [56491.47205181],
           [58862.72469988],
           [57281.88960117],
           [57281.88960117],
           [57281.88960117],
           [59653.14224924],
           [62024.39489731],
           [62024.39489731],
           [59653.14224924],
           [59653.14224924],
           [59653.14224924],
           [59653.14224924],
           [58072.30715053],
           [58072.30715053],
           [65976.48264409],
           [62024.39489731],
           [62024.39489731],
           [66766.90019345],
           [63605.22999602],
           [62814.81244667],
           [57281.88960117],
           [60443.5597986 ],
           [62814.81244667],
           [62024.39489731],
           [60443.5597986 ],
           [55701.05450246],
           [68347.73529216],
           [67557.31774281],
           [67557.31774281],
           [66766.90019345],
           [62814.81244667],
           [62814.81244667],
           [66766.90019345],
           [66766.90019345],
           [65976.48264409],
           [74671.07568702],
           [74671.07568702],
           [64395.64754538],
           [64395.64754538],
           [64395.64754538],
           [64395.64754538],
           [60443.5597986 ],
           [60443.5597986 ],
           [61233.97734795],
           [61233.97734795],
           [61233.97734795],
           [61233.97734795],
           [61233.97734795],
           [61233.97734795],
           [64395.64754538],
           [63605.22999602],
           [62814.81244667],
           [62814.81244667],
           [62814.81244667],
           [56491.47205181],
           [56491.47205181],
           [56491.47205181],
           [56491.47205181],
           [73880.65813766],
           [64395.64754538],
           [73880.65813766],
           [64395.64754538],
           [64395.64754538],
           [70718.98794023],
           [62814.81244667],
           [60443.5597986 ],
           [60443.5597986 ],
           [56491.47205181],
           [67557.31774281],
           [62024.39489731],
           [59653.14224924],
           [59653.14224924],
           [59653.14224924],
           [59653.14224924],
           [54910.6369531 ],
           [54910.6369531 ],
           [59653.14224924],
           [57281.88960117],
           [55701.05450246],
           [58862.72469988],
           [57281.88960117]])




```python
#Multiple Linear Model Estimator (SLM)
import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import * #<--Import the method


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substmean('horsepower')

converttoint('price')
substmean('price')

substmean('curb-weight')
substmean('highway-mpg')


#create a LinearRegression() object:
lm = LinearRegression()

#Define variables:
Z = df[['highway-mpg','engine-size','curb-weight','horsepower']]
y = df[['price']]#<-------- USE [[]] TO make it 2D instead of 1D array

#fit the model/find b0 and b1
lm.fit(Z,y)

#See the predictor's prediction changed values:
yhat = lm.predict(Z)

#calculate intercept (b0)
b0=lm.intercept_

#calculate slope (b1)
b1 = lm.coef_

#Visualize Results
print('predictions: ',yhat,'b0 (interceptor): ', b0, 'b1 (slope): ',b1)

#formula to predict a target value ('price') using one predictor (highway-mpg) as a trainer to fit:
#----------y=b0+b1*X+b2*X2---(price_estimated = lm.intercept+lm.coef_1*'highway-mpg'+lm.coef_1*'highway-mpg')
price_estimated= -9082.93177067+68.0125147* df['highway-mpg']+101.45594188 * df['engine-size']+3.78514218 * df['curb-weight']+17.5506254* df['horsepower'] 
price_estimated
```

    predictions:  [[ 1.38626645e+04]
     [ 1.79582987e+04]
     [ 1.05714315e+04]
     [ 1.59263644e+04]
     [ 1.44346837e+04]
     [ 1.57102766e+04]
     [ 1.61266422e+04]
     [ 1.69855826e+04]
     [ 1.70756604e+04]
     [ 1.07399757e+04]
     [ 1.07399757e+04]
     [ 1.80328533e+04]
     [ 1.82410361e+04]
     [ 1.95427649e+04]
     [ 2.60453078e+04]
     [ 2.66130791e+04]
     [ 2.72222469e+04]
     [-2.40610134e+01]
     [ 5.44546509e+03]
     [ 5.57794506e+03]
     [ 5.55395915e+03]
     [ 5.75799670e+03]
     [ 8.66432144e+03]
     [ 6.10244463e+03]
     [ 6.18571776e+03]
     [ 6.18571776e+03]
     [ 8.90278540e+03]
     [ 1.23941082e+04]
     [ 1.82967702e+04]
     [ 4.08022391e+03]
     [ 5.88556048e+03]
     [ 4.08190573e+03]
     [ 6.61561274e+03]
     [ 6.67617502e+03]
     [ 6.88057269e+03]
     [ 6.93356468e+03]
     [ 9.80574055e+03]
     [ 1.00063531e+04]
     [ 1.00631302e+04]
     [ 1.03205199e+04]
     [ 1.12758601e+04]
     [ 1.04032274e+04]
     [ 1.04211409e+04]
     [ 5.44546509e+03]
     [ 5.57794506e+03]
     [ 1.29460974e+04]
     [ 3.42797616e+04]
     [ 3.42797616e+04]
     [ 4.23850680e+04]
     [ 6.38853223e+03]
     [ 5.95029605e+03]
     [ 5.96922176e+03]
     [ 6.12062745e+03]
     [ 6.13955316e+03]
     [ 7.23594787e+03]
     [ 7.23594787e+03]
     [ 7.25487359e+03]
     [ 9.30144562e+03]
     [ 1.16201093e+04]
     [ 1.17147379e+04]
     [ 1.16201093e+04]
     [ 1.17147379e+04]
     [ 1.08085099e+04]
     [ 1.17715150e+04]
     [ 1.54969669e+04]
     [ 1.33432053e+04]
     [ 2.32466944e+04]
     [ 2.41362028e+04]
     [ 2.31709916e+04]
     [ 2.42119057e+04]
     [ 3.03103121e+04]
     [ 3.01021292e+04]
     [ 3.90686677e+04]
     [ 3.79625926e+04]
     [ 1.75747229e+04]
     [ 5.91584701e+03]
     [ 6.21829825e+03]
     [ 6.44540678e+03]
     [ 8.72866886e+03]
     [ 1.10435059e+04]
     [ 1.14745587e+04]
     [ 1.83800433e+04]
     [ 1.87131358e+04]
     [ 1.87320615e+04]
     [ 1.16146090e+04]
     [ 1.17660146e+04]
     [ 1.11684156e+04]
     [ 1.11684156e+04]
     [ 6.60295828e+03]
     [ 6.56632068e+03]
     [ 6.71272740e+03]
     [ 6.78843024e+03]
     [ 7.11395247e+03]
     [ 6.83763709e+03]
     [ 7.12909304e+03]
     [ 6.91333994e+03]
     [ 7.16315932e+03]
     [ 7.05339020e+03]
     [ 1.12784368e+04]
     [ 1.11951637e+04]
     [ 2.21670285e+04]
     [ 2.29278421e+04]
     [ 2.18305110e+04]
     [ 2.20125525e+04]
     [ 2.31079923e+04]
     [ 2.22699422e+04]
     [ 1.45930209e+04]
     [ 1.78623674e+04]
     [ 1.53879008e+04]
     [ 1.92884056e+04]
     [ 1.47661025e+04]
     [ 1.80705502e+04]
     [ 1.55609824e+04]
     [ 1.94965884e+04]
     [ 1.48012038e+04]
     [ 1.80705502e+04]
     [ 1.72195479e+04]
     [ 5.71293512e+03]
     [ 8.66432144e+03]
     [ 6.10244463e+03]
     [ 6.18571776e+03]
     [ 7.76196402e+03]
     [ 1.23941082e+04]
     [ 1.83232662e+04]
     [ 1.74254420e+04]
     [ 2.29640394e+04]
     [ 2.29640394e+04]
     [ 2.31305856e+04]
     [ 2.74036427e+04]
     [ 1.37704607e+04]
     [ 1.33200288e+04]
     [ 1.32803635e+04]
     [ 1.34204138e+04]
     [ 1.34658355e+04]
     [ 1.36588777e+04]
     [ 1.48616911e+04]
     [ 1.50093117e+04]
     [ 7.28037868e+03]
     [ 9.07161907e+03]
     [ 9.52583613e+03]
     [ 8.91612817e+03]
     [ 9.35850962e+03]
     [ 1.02049010e+04]
     [ 1.06407125e+04]
     [ 1.13507733e+04]
     [ 9.80503636e+03]
     [ 1.07082048e+04]
     [ 1.05011424e+04]
     [ 1.22887683e+04]
     [ 6.20017281e+03]
     [ 6.47636814e+03]
     [ 6.38173959e+03]
     [ 7.45281478e+03]
     [ 7.83072878e+03]
     [ 1.09345454e+04]
     [ 7.44871214e+03]
     [ 7.55469612e+03]
     [ 9.22280479e+03]
     [ 8.47466713e+03]
     [ 6.81779384e+03]
     [ 7.80794052e+03]
     [ 7.87607308e+03]
     [ 7.98584220e+03]
     [ 8.11832217e+03]
     [ 9.42640469e+03]
     [ 9.55888466e+03]
     [ 1.53393940e+04]
     [ 1.53242534e+04]
     [ 1.53810305e+04]
     [ 1.58655287e+04]
     [ 1.59980087e+04]
     [ 1.69859308e+04]
     [ 1.14011659e+04]
     [ 1.05011571e+04]
     [ 1.18702834e+04]
     [ 1.18702834e+04]
     [ 1.20368297e+04]
     [ 2.07239678e+04]
     [ 2.08753734e+04]
     [ 2.12229117e+04]
     [ 2.02840551e+04]
     [ 7.10055790e+03]
     [ 9.51652263e+03]
     [ 7.11191333e+03]
     [ 9.52787805e+03]
     [ 9.76634201e+03]
     [ 7.87295621e+03]
     [ 1.02602550e+04]
     [ 1.01146697e+04]
     [ 9.98976003e+03]
     [ 1.50856081e+04]
     [ 9.12914324e+03]
     [ 1.11131524e+04]
     [ 1.63411109e+04]
     [ 1.68028983e+04]
     [ 1.64281692e+04]
     [ 1.68331794e+04]
     [ 1.69790246e+04]
     [ 1.74029605e+04]
     [ 1.64925166e+04]
     [ 1.78710417e+04]
     [ 2.06572904e+04]
     [ 1.78290106e+04]
     [ 1.71129198e+04]] b0 (interceptor):  [-9082.93177067] b1 (slope):  [[-68.0125147  101.45594188   3.78514218  17.5506254 ]]
    




    0      17535.340265
    1      21494.949463
    2      14652.182401
    3      18918.915086
    4      17835.309432
               ...     
    199    20301.217457
    200    21271.667473
    201    23785.866062
    202    21501.686384
    203    20513.545553
    Length: 204, dtype: float64




```python
#Regression Plot to Evaluate Linear Model Estimator (SLM) 
import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import * #<--Import the method


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substmean('horsepower')

converttoint('price')
substmean('price')

#create a LinearRegression() object:
lm = LinearRegression()

#Define variables:
X = df[['horsepower']]
y = df[['price']]#<-------- USE [[]] TO make it 2D instead of 1D array

#fit the model/find b0 and b1
lm.fit(X,y)

#See the predictor's prediction changed values:
yhat = lm.predict(X)

#calculate intercept (b0)
b0=lm.intercept_

#calculate slope (b1)
b1 = lm.coef_

#formula to predict a target value ('price') using one predictor (highway-mpg) as a trainer to fit:
#----------y=b0-b1*X---(price predicted = lm.intercept_-lm.coef_*'highway-mpg')
price_predicted=b0+b1*X
# price_predicted

# Create the Regression Plot
# sns.regplot(x=df['horsepower'], y=df['price'])
# plt.ylim(0,)

#Check results with a Residual PLot
sns.residplot(x=df['horsepower'],y=df['price'])
#NOTE: The residual plot shows the values spreading along the x-axis, it means that the linear model is not acurate, so try a multilinear model.
```




    <AxesSubplot:xlabel='horsepower', ylabel='price'>




    
![png](output_29_1.png)
    



```python
#Regression Plot to Evaluate Multi Linear Model Estimator (SLM) 
import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import * #<--Import the method


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substmean('horsepower')

converttoint('price')
substmean('price')

#create a LinearRegression() object:
lm = LinearRegression()

#Define variables:
Z = df[['highway-mpg','engine-size','curb-weight','horsepower']]
y = df[['price']]#<-------- USE [[]] TO make it 2D instead of 1D array

#fit the model/find b0 and b1
lm.fit(Z,y)

#See the predictor's prediction changed values:
yhat = lm.predict(Z)

#calculate intercept (b0)
b0=lm.intercept_

#calculate slope (b1)
b1 = lm.coef_

#Visualize Results
# print('predictions: ',yhat,'b0 (interceptor): ', b0, 'b1 (slope): ',b1)


# NOTE: regplot only takes 1d arrays so multi linear models can not be graphed with regplot.

#Compare predictions and actual value with a Distribution Plot
axl=sns.distplot(y,hist=False,color='r',label='Price')
sns.distplot(yhat,hist=False,color='b', label='Predicted Values',ax=axl, axlabel='Price')

# NOTE: Boyj graphs are similar which means its more presice

```

    C:\ProgramData\Anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
      warnings.warn(msg, FutureWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
      warnings.warn(msg, FutureWarning)
    




    <AxesSubplot:xlabel='Price', ylabel='Density'>




    
![png](output_30_2.png)
    



```python
import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import * #<--Import the method
from sklearn.preprocessing import StandardScaler


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substituetomean('horsepower')

converttoint('price')
substituetomean('price')

SCALE = StandardScaler()
SCALE.fit(df[['horsepower','highway-mpg']])

x_scale=SCALE.transform(df[['horsepower','highway-mpg']])
# df[['horsepower','highway-mpg']]
x_scale
```




    array([[ 0.17180685, -0.54779597],
           [ 1.25995825, -0.69311506],
           [-0.05594577, -0.11183871],
           [ 0.27303024, -1.27439141],
           [ 0.146501  , -0.83843415],
           [ 0.146501  , -0.83843415],
           [ 0.146501  , -0.83843415],
           [ 0.9056764 , -1.56502959],
           [ 1.41179333, -1.27439141],
           [-0.08125161, -0.2571578 ],
           [-0.08125161, -0.2571578 ],
           [ 0.42486531, -0.40247689],
           [ 0.42486531, -0.40247689],
           [ 0.42486531, -0.83843415],
           [ 1.96852195, -1.27439141],
           [ 1.96852195, -1.27439141],
           [ 1.96852195, -1.56502959],
           [-1.42246148,  3.23050032],
           [-0.86573286,  1.77730943],
           [-0.86573286,  1.77730943],
           [-0.91634455,  1.48667126],
           [-0.91634455,  1.05071399],
           [-0.05594577, -0.11183871],
           [-0.91634455,  1.05071399],
           [-0.91634455,  1.05071399],
           [-0.91634455,  1.05071399],
           [-0.05594577, -0.11183871],
           [-0.41022762, -0.11183871],
           [ 1.03220563, -0.98375324],
           [-1.16940301,  3.3758194 ],
           [-0.71389778,  1.05071399],
           [-1.11879132,  1.63199035],
           [-0.71389778,  0.46943764],
           [-0.71389778,  0.46943764],
           [-0.71389778,  0.46943764],
           [-0.71389778,  0.46943764],
           [-0.46083931,  0.32411855],
           [-0.46083931,  0.32411855],
           [-0.46083931,  0.32411855],
           [-0.46083931,  0.32411855],
           [-0.08125161, -0.40247689],
           [-0.10655746,  0.03348038],
           [-0.66328608, -0.2571578 ],
           [-0.86573286,  1.77730943],
           [-0.86573286,  1.77730943],
           [-0.35961593, -0.2571578 ],
           [ 1.81668687, -1.71034868],
           [ 1.81668687, -1.71034868],
           [ 3.99298967, -2.00098685],
           [-0.91634455,  0.03348038],
           [-0.91634455,  1.05071399],
           [-0.91634455,  1.05071399],
           [-0.91634455,  1.05071399],
           [-0.91634455,  1.05071399],
           [-0.08125161, -1.12907233],
           [-0.08125161, -1.12907233],
           [-0.08125161, -1.12907233],
           [ 0.77914717, -1.12907233],
           [-0.51145101,  0.17879947],
           [-0.51145101,  0.17879947],
           [-0.51145101,  0.17879947],
           [-0.51145101,  0.17879947],
           [-1.01756793,  1.63199035],
           [-0.51145101,  0.17879947],
           [ 0.39955947, -0.54779597],
           [-0.81512116,  1.19603308],
           [ 0.47547701, -0.83843415],
           [ 0.47547701, -0.83843415],
           [ 0.47547701, -0.83843415],
           [ 0.47547701, -0.83843415],
           [ 1.2852641 , -1.85566777],
           [ 1.2852641 , -1.85566777],
           [ 2.01913364, -2.14630594],
           [ 2.01913364, -2.14630594],
           [ 1.79138102, -0.98375324],
           [-0.91634455,  1.48667126],
           [-0.91634455,  1.05071399],
           [-0.91634455,  1.05071399],
           [-0.05594577, -0.11183871],
           [ 0.29833608, -0.11183871],
           [-0.41022762,  0.17879947],
           [ 1.03220563, -0.98375324],
           [ 1.03220563, -0.98375324],
           [ 1.03220563, -0.98375324],
           [-0.41022762,  0.17879947],
           [-0.41022762,  0.17879947],
           [ 0.29833608, -0.11183871],
           [ 0.29833608, -0.11183871],
           [-0.8910387 ,  0.90539491],
           [-1.24532055,  2.79454305],
           [-0.8910387 ,  0.90539491],
           [-0.8910387 ,  0.90539491],
           [-0.8910387 ,  0.90539491],
           [-0.8910387 ,  0.90539491],
           [-0.8910387 ,  0.90539491],
           [-0.8910387 ,  0.90539491],
           [-0.8910387 ,  0.90539491],
           [-0.8910387 ,  0.90539491],
           [-0.182475  ,  0.46943764],
           [-0.182475  ,  0.46943764],
           [ 1.20934656, -1.27439141],
           [ 1.20934656, -1.27439141],
           [ 1.20934656, -0.83843415],
           [ 1.41179333, -0.83843415],
           [ 2.42402719, -1.12907233],
           [ 1.41179333, -0.83843415],
           [-0.182475  , -0.98375324],
           [-0.23308669,  0.32411855],
           [-0.182475  , -0.98375324],
           [-0.23308669, -0.83843415],
           [-0.23308669, -0.98375324],
           [-0.23308669,  0.32411855],
           [-0.23308669, -0.98375324],
           [-0.23308669, -0.83843415],
           [-0.182475  , -0.98375324],
           [-0.23308669,  0.32411855],
           [ 0.95628809, -0.98375324],
           [-0.91634455,  1.48667126],
           [-0.05594577, -0.11183871],
           [-0.91634455,  1.05071399],
           [-0.91634455,  1.05071399],
           [-0.91634455,  1.05071399],
           [-0.41022762, -0.11183871],
           [ 1.03220563, -0.98375324],
           [ 0.98159394, -0.54779597],
           [ 2.60116811, -0.83843415],
           [ 2.60116811, -0.83843415],
           [ 2.60116811, -0.83843415],
           [ 4.65094168, -0.40247689],
           [-0.03063992,  0.03348038],
           [-0.03063992,  0.03348038],
           [ 0.146501  , -0.40247689],
           [ 0.146501  , -0.40247689],
           [ 0.146501  , -0.40247689],
           [ 0.146501  , -0.40247689],
           [ 1.41179333, -0.69311506],
           [ 1.41179333, -0.69311506],
           [-0.8910387 ,  0.76007582],
           [-0.78981532,  0.03348038],
           [-0.78981532,  0.03348038],
           [-0.5620627 ,  0.90539491],
           [-0.5620627 ,  0.32411855],
           [-0.25839254,  0.17879947],
           [-0.5620627 , -0.83843415],
           [ 0.17180685, -0.2571578 ],
           [-0.5620627 ,  0.17879947],
           [-0.25839254,  0.03348038],
           [-0.5620627 , -0.2571578 ],
           [ 0.17180685, -1.12907233],
           [-1.06817963,  1.19603308],
           [-1.06817963,  1.05071399],
           [-1.06817963,  1.05071399],
           [-1.06817963,  0.90539491],
           [-1.06817963,  0.17879947],
           [-1.06817963,  0.17879947],
           [-0.86573286,  0.90539491],
           [-0.86573286,  0.90539491],
           [-1.22001471,  0.76007582],
           [-1.22001471,  2.35858579],
           [-0.86573286,  2.35858579],
           [-0.86573286,  0.46943764],
           [-0.86573286,  0.46943764],
           [-0.86573286,  0.46943764],
           [-0.86573286,  0.46943764],
           [ 0.1971127 , -0.2571578 ],
           [ 0.1971127 , -0.2571578 ],
           [ 0.29833608, -0.11183871],
           [ 0.29833608, -0.11183871],
           [ 0.29833608, -0.11183871],
           [ 0.29833608, -0.11183871],
           [ 0.29833608, -0.11183871],
           [ 0.29833608, -0.11183871],
           [-0.30900423,  0.46943764],
           [-0.78981532,  0.32411855],
           [-0.30900423,  0.17879947],
           [-0.30900423,  0.17879947],
           [-0.30900423,  0.17879947],
           [ 1.43709917, -0.98375324],
           [ 1.43709917, -0.98375324],
           [ 1.31056994, -0.98375324],
           [ 1.31056994, -0.98375324],
           [-1.32123809,  2.2132667 ],
           [-0.48614516,  0.46943764],
           [-1.32123809,  2.2132667 ],
           [-0.48614516,  0.46943764],
           [-0.48614516,  0.46943764],
           [-0.91634455,  1.63199035],
           [-0.10655746,  0.17879947],
           [-0.35961593, -0.2571578 ],
           [-0.35961593, -0.2571578 ],
           [ 0.146501  , -0.98375324],
           [-0.91634455,  1.05071399],
           [-0.41022762,  0.03348038],
           [ 0.24772439, -0.40247689],
           [ 0.24772439, -0.40247689],
           [ 0.24772439, -0.40247689],
           [ 0.24772439, -0.40247689],
           [ 1.46240502, -1.27439141],
           [ 1.46240502, -1.27439141],
           [ 0.24772439, -0.40247689],
           [ 1.41179333, -0.83843415],
           [ 0.75384132, -1.12907233],
           [ 0.04527762, -0.54779597],
           [ 0.24772439, -0.83843415]])




```python
#Pipelines to normalization

import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import * #<--Import the methods
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substituetomean('horsepower')

converttoint('price')
substituetomean('price')

SCALE = StandardScaler()
SCALE.fit(df[['horsepower','highway-mpg']])

x_scale=SCALE.transform(df[['horsepower','highway-mpg']])

Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree=2) ), ('model', LinearRegression())]
pipe = Pipeline(Input)

#Train the pipeline
pipe.train(X['horsepower','curb-weight','engine-size','highway-mpg'],y)
yhat = pipe.predict(X['horsepower','curb-weight','engine-size','highway-mpg'],y)
yhat
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_3540\1254422845.py in <module>
         50 
         51 #Train the pipeline
    ---> 52 pipe.train(X['horsepower','curb-weight','engine-size','highway-mpg'],y)
         53 yhat = pipe.predict(df['horsepower','curb-weight','engine-size','highway-mpg'],y)
         54 yhat
    

    AttributeError: 'Pipeline' object has no attribute 'train'



```python
#Measure For in-sample Evaluation:
#Mean Squared Error (MSE)

import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import * #<--Import the methods
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substituetomean('horsepower')

converttoint('price')
substituetomean('price')

#create a LinearRegression() object:
lm = LinearRegression()

#Define variables:
Z = df[['highway-mpg','engine-size','curb-weight','horsepower']]
y = df[['price']]#<-------- USE [[]] TO make it 2D instead of 1D array


#fit the model/find b0 and b1
lm.fit(Z,y)

#See the predictor's prediction changed values:
yhat = lm.predict(Z)

#calculate intercept (b0)
b0=lm.intercept_

#calculate slope (b1)
b1 = lm.coef_

#MSE
#fit the inputs to the MSE (target and predicted target)

mean_squared_error(df['price'],yhat)
```




    13862438.941860871




```python
#Measure For in-sample Evaluation:
#R^2 Coefficient of regretion

import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import * #<--Import the methods
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substituetomean('horsepower')

converttoint('price')
substituetomean('price')

#create a LinearRegression() object:
lm = LinearRegression()

#Define variables:
Z = df[['highway-mpg','engine-size','curb-weight','horsepower']]
y = df[['price']]#<-------- USE [[]] TO make it 2D instead of 1D array


#fit the model/find b0 and b1
lm.fit(Z,y)

#See the predictor's prediction changed values:
yhat = lm.predict(Z)

#calculate intercept (b0)
b0=lm.intercept_

#calculate slope (b1)
b1 = lm.coef_

#R^2 Coefficient of regretion
r2 = r2_score(y,yhat)
r2
```




    0.7761176218312253




```python
#Making sure the model is correct: for a Simple linear model:

import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import * #<--Import the methods
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substituetomean('horsepower')

converttoint('price')
substituetomean('price')

#MOdel:
lm = LinearRegression() #create a LinearRegression() object:

#Define variables:
y = df[['price']]#<-------- USE [[]] TO make it 2D instead of 1D array
Z = df[['highway-mpg']]

lm.fit(Z,y)#fit the model/find b0 and b1

yhat = lm.predict(Z)#See the predictor's prediction changed values:

# b0=lm.intercept_ #calculate intercept (b0)

# b1 = lm.coef_#calculate slope (b1)

#Making sure the model is correct:
lm.fit(df[['highway-mpg']],df[['price']])#<--train the model with the variables (to predict price deppending on 'highway-mpg' )
verification = lm.predict([[30]])#<-- try a prediction using a single value from the predictor variable
b1= lm.coef_ #<-- check the coeffisient of the predictor to the target
b0= lm.intercept_#<--check the intercept point value
newprice = b0-790.418*df[['highway-mpg']]#<--Apply the simple linear regresion formula (substitute b1 it's int value)
# verification
# newprice
# df['highway-mpg']
# b1
# df['price']

#NOTE: As such, an increase of one unit in highway-mpg, the value of the car decreases approximately 821 dollars; this value also seems reasonable.

#Use a plot to visualize the distribution:
meanvalue = df['highway-mpg'].mean()
df['highway-mpg'].replace(0, meanvalue, inplace=True)

# sns.regplot(x=df['highway-mpg'],y=df['price'] )
# plt.ylim(0,)

#Then using a residual plot to see data related to the output distribution (to see how linear is the relatuonship of the values on the model)
# sns.residplot(x=df['horsepower'],y=df['price'])

#NOTE: too many negative values on the residual plot means a non linear relationship, so the model needs more predictor data 

#Distribution plot:
# yhat = lm.predict(df[['horsepower']])
axl = sns.distplot(df['price'], hist=False, color='r', label='Price')
sns.distplot(yhat, hist = False, color='b', label= 'Adapted Values')
#NOTE: in the Distplot the values from 30000 to 50000 are inacurate, this confirms that a multiple linear model would be more acurate
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\base.py:450: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
    C:\ProgramData\Anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
      warnings.warn(msg, FutureWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `kdeplot` (an axes-level function for kernel density plots).
      warnings.warn(msg, FutureWarning)
    




    <AxesSubplot:xlabel='price', ylabel='Density'>




    
![png](output_35_2.png)
    



```python
#Model Evaluation with Split the data to train and test 

import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import * #<--Import the methods
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substituetomean('horsepower')

converttoint('price')
substituetomean('price')

#create a LinearRegression() object:
lm = LinearRegression()
#Define variables:
Z = df[['highway-mpg','engine-size','curb-weight','horsepower']]
y = df[['price']]
#fit the model/find b0 and b1
lm.fit(Z,y)
#See the predictor's prediction changed values:
yhat = lm.predict(Z)
#calculate intercept (b0)
b0=lm.intercept_
#calculate slope (b1)
b1 = lm.coef_

#Split the data:
x_data = Z
y_data = y
x_train, y_train, x_test,y_test = train_test_split(x_data,y_data, test_size=30, random_state=0)
x_train, y_train, x_test,y_test
#The result are two arrays, one (the smaller one) is for the testing (30% of the data), and the big array is the training data (70%)
```




    (     highway-mpg  engine-size  curb-weight  horsepower
     203           25          141         3062         114
     60            32          122         2385          84
     16            20          209         3505         182
     74            24          140         2910         175
     80            32          122         2328          88
     ..           ...          ...          ...         ...
     67            25          183         3750         123
     192           31          109         2563          88
     117           41           90         1918          68
     47            19          258         4066         176
     172           34          122         2326          92
     
     [174 rows x 4 columns],
          highway-mpg  engine-size  curb-weight  horsepower
     18            43           90         1874          70
     45            29          119         2734          90
     33            34           92         1956          76
     37            33          110         2289          86
     109           25          152         3430          95
     90            37           97         1918          69
     5             25          136         2844         110
     124           27          151         2778         143
     12            28          164         2765         121
     153           32           92         2290          62
     61            32          122         2410          84
     187           32          109         2300         100
     166           30          146         2540         116
     160           34           98         2122          70
     155           37           98         2081          70
     7             20          131         3086         140
     129           31          132         2579         103
     131           28          121         2658         110
     75            41           92         1918          68
     66            25          183         3515         123
     44            43           90         1909          70
     146           31          108         2455          94
     63            32          122         2425          84
     135           26          121         2808         160
     86            30          110         2403         116
     141           33          108         2190          82
     188           29          109         2254          90
     182           34          109         2209          85
     161           34           98         2140          70
     96            37           97         2037          69,
          price
     203  22625
     60   10595
     16   36880
     74   16503
     80    8499
     ..     ...
     67   28248
     192  12290
     117   5572
     47   35550
     172   8948
     
     [174 rows x 1 columns],
          price
     18    6295
     45   11048
     33    7129
     37    9095
     109  13860
     90    6649
     5    17710
     124  22018
     12   21105
     153   7898
     61   10245
     187   9995
     166   8449
     160   8358
     155   6938
     7    23875
     129   9295
     131  11850
     75    5389
     66   25552
     44   12946
     146  10198
     63   11245
     135  18150
     86    9279
     141   7775
     188  11595
     182   7975
     161   9258
     96    7999)




```python
#Cross Validation

# import pandas as pd
# import seaborn as sns
# import numpy as np
# import scipy
# from scipy import *
# import matplotlib.pyplot as plt
# from sklearn.linear_model import * #<--Import the methods
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substituetomean('horsepower')

converttoint('price')
substituetomean('price')

#create a LinearRegression() object:
lm = LinearRegression()
#Define variables:
Z = df[['highway-mpg','engine-size','curb-weight','horsepower']]
y = df[['price']]
#fit the model/find b0 and b1
lm.fit(Z,y)
#See the predictor's prediction changed values:
yhat = lm.predict(Z)
#calculate intercept (b0)
b0=lm.intercept_
#calculate slope (b1)
b1 = lm.coef_

#Split the data:
x_data = Z
y_data = y
x_train, y_train, x_test,y_test = train_test_split(x_data,y_data, test_size=30, random_state=0)
x_train, y_train, x_test,y_test
#The result are two arrays, one (the smaller one) is for the testing (30% of the data), and the big array is the training data (70%)
#cross validation:
score = cross_val_score(lm, x_data, y_data, cv=3)
# np.mean(score)

#see the prediction for each element on the score's array

# yhat2= cross_val_predict(lm,x_data, y_data, cv=3 )
# yhat2
score
```




    array([0.75644827, 0.68685635, 0.57472808])




```python
#Ridge Regression

import pandas as pd
import seaborn as sns
import numpy as np
import scipy
from scipy import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substituetomean('horsepower')

converttoint('price')
substituetomean('price')

#create a LinearRegression() object:
lm = LinearRegression()
#Define variables:
Z = df[['highway-mpg','engine-size','curb-weight','horsepower']]
y = df[['price']]

ridge = Ridge(alpha=0.1) #<--create an object to ridge model
ridge.fit(Z,y) #<-fit the model
yhat=ridge.predict(Z) #<--predict the model data
df.shape

```




    (204, 26)




```python
#Ridge Regression

import pandas as pd
import seaborn as sns
import numpy as np
import scipy
from scipy import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import *
data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substituetomean('horsepower')

converttoint('price')
substituetomean('price')

#create a LinearRegression() object:
lm = LinearRegression()
#Define variables:
Z = df[['highway-mpg','engine-size','curb-weight','horsepower']]
y = df[['price']]

parameters1 = [{'alpha':[0.001,0.1,1,10,100,1000,10000,100000,1000000]}]
RR=Ridge()
Grid1=GridSearchCV(RR,parameters1,cv=4)
Grid1.fit(Z,y)
Grid1.best_estimator_

scores=Grid1.cv_results_
scores['mean_test_score']
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_3540\2866932721.py in <module>
         45 parameters1 = [{'alpha':[0.001,0.1,1,10,100,1000,10000,100000,1000000]}]
         46 RR=Ridge()
    ---> 47 Grid1=GridSearchCV(RR,parameters1,cv=4)
         48 Grid1.fit(Z,y)
         49 Grid1.best_estimator_
    

    NameError: name 'GridSearchCV' is not defined



```python
#Regression Plot to Evaluate Linear Model Estimator (SLM) 
import pandas as pd
import seaborn as sns
import scipy
from scipy import *
import matplotlib.pyplot as plt
from sklearn.linear_model import * #<--Import the method


data = ("imports-85 (1).csv")

df = pd.read_csv(data)
headers = ['symboling', 'normalized-losses','make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
           'drive-wheels', 'engine-location', 'weel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinderss','engine-size','fuel-system','bore','stroke','compression-ratio', 'horsepower',
           'peak-rpm','city-mpg','highway-mpg','price']

df.columns=headers

#function to convert from 'object' to "int" and replace its NAN/'?' values for its mean value:
def converttoint(n):
    df[n].replace("?", 0, inplace=True)
    df[n]=df[n].astype("int")
    
def substituetomean(l):
    meanvalue = df[l].mean()                        
    df[l].replace(0, meanvalue, inplace=True)
    df[l]=df[l].astype("int")
    

#convert to 'int' non-numeric variables:  
converttoint('horsepower')
substituetomean('horsepower')

converttoint('price')
substituetomean('price')
x=(10,100)
# c= df.corr()
x.corr()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_15236\2029857682.py in <module>
         37 x=(10,100)
         38 # c= df.corr()
    ---> 39 x.corr()
    

    AttributeError: 'tuple' object has no attribute 'corr'



```python

```
