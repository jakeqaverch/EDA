
# Dataset Exploration and Analysis

This project explores the dataset through the following steps:

## 1. Import Libraries and Dataset
```sh
# import libraries and set options
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as pl
import datetime as dt
import numpy as np
import math
import seaborn as sns
import networkx as nx
pd.set_option('display.max_rows', 100)
```
 
```sh
# import dataset
data = pd.read_csv(
    "/content/mydata - mydata.csv",  # Path to the data - CSV file
    dtype={  # Specify data types for specific columns
        "gender": "category",      # Gender as a categorical variable
        'managerid': 'string',     # Manager ID as a string
        "race": "category",        # Race as a categorical variable
        "level": "category",       # Level as a categorical variable
        "levelnumber": "int64",    # Level number as an integer
        "city": "category",        # City as a categorical variable
        "department": "category"   # Department as a categorical variable
    },
    index_col='id',                # Set the 'id' column as the index
    parse_dates=['startdate'],     # Parse the 'startdate' column as dates
    date_format='%Y-%m-%d'         # Specify the date format for 'startdate'
).sort_index()                     # Sort the DataFrame by index
```


## 2. Data Overview and Data Cleaning 
```sh
# check the overview of the data
data.info() 
```

![image](https://github.com/Hello-DataSkillUp/Jake_EDA1/assets/165890395/4dde8ad6-68ef-41a5-a990-68249c8c9015)


- The dataset has 13 columns and 1000 entries. 
- Each columns data type is identified. 
- Every column except manager ID has 1000 non - null values, indicating almost no missing data. 


```sh
# get count and proportion for all categorical variables
for i in data.select_dtypes(['category']):
    normed = pd.DataFrame(data[i].value_counts(normalize = False))  # gives counts
    nonnormed = pd.DataFrame(data[i].value_counts(normalize = True))  # gives proportions
    result = pd.concat([normed, nonnormed], axis=1)  # combine results into a single dataframe
    print(result)
```

| Gender | Count | Proportion |
|--------|-------|------------|
| Male   | 510   | 0.51       |
| Female | 490   | 0.49       |

| Race              | Count | Proportion |
|-------------------|-------|------------|
| Caucasian         | 543   | 0.543      |
| Asian             | 228   | 0.228      |
| Hispanic          | 96    | 0.096      |
| African American  | 90    | 0.090      |
| Other             | 43    | 0.043      |

| Department               | Count | Proportion |
|--------------------------|-------|------------|
| Support                  | 96    | 0.096      |
| Product Management       | 94    | 0.094      |
| Sales                    | 94    | 0.094      |
| Services                 | 90    | 0.090      |
| Engineering              | 88    | 0.088      |
| Legal                    | 84    | 0.084      |
| Business Development     | 81    | 0.081      |
| Marketing                | 81    | 0.081      |
| Accounting               | 75    | 0.075      |
| Human Resources          | 73    | 0.073      |
| Training                 | 72    | 0.072      |
| Research and Development | 71    | 0.071      |
| Admin                    | 1     | 0.001      |

| Level        | Count | Proportion |
|--------------|-------|------------|
| Entry Level  | 424   | 0.424      |
| Junior       | 360   | 0.360      |
| Mid-Level    | 126   | 0.126      |
| Senior       | 59    | 0.059      |
| VP           | 19    | 0.019      |
| Executive    | 11    | 0.011      |
| CEO          | 1     | 0.001      |

| City          | Count | Proportion |
|---------------|-------|------------|
| San Francisco | 391   | 0.391      |
| New York      | 246   | 0.246      |
| Chicago       | 191   | 0.191      |
| Boston        | 88    | 0.088      |
| Remote        | 84    | 0.084      |


- Gender distribution is balanced
- Majority of the individuals are Caucasian (54.3% of the dataset), followed by Asians (22.8%)
- Significant portion of the workforce is at the Entry Level (42.4%) and Junior level (36%), indicating a young or early-career workforce. Senior levels make up a smaller fraction, which is typical as organizational pyramids narrow at the top
-  Majority of employees are based in San Francisco and New York. Interestingly, we have some remote employees as well

```sh
# generate and print descriptive statistics for numeric variables
for x in data.select_dtypes(['int64', 'float64']):
    print(pd.DataFrame(data[x].describe()))
```

#### Descriptive Statistics for Level Number

| Statistic | Value      |
|-----------|------------|
| Count     | 1000.000   |
| Mean      | 1.927      |
| Std Dev   | 1.079      |
| Min       | 1.000      |
| 25%       | 1.000      |
| 50% (Median) | 2.000   |
| 75%       | 2.000      |
| Max       | 8.000      |

#### Descriptive Statistics for Salary

| Statistic | Value      |
|-----------|------------|
| Count     | 1000.000   |
| Mean      | $84,360.09 |
| Std Dev   | $44,728.31 |
| Min       | $25,002.00 |
| 25%       | $56,528.75 |
| 50% (Median) | $72,749.00 |
| 75%       | $93,815.00 |
| Max       | $288,502.00 |

#### Descriptive Statistics for Bonus

| Statistic | Value      |
|-----------|------------|
| Count     | 1000.000   |
| Mean      | $10,777.31 |
| Std Dev   | $15,545.65 |
| Min       | $1,132.43  |
| 25%       | $2,994.71  |
| 50% (Median) | $6,410.39 |
| 75%       | $9,663.54  |
| Max       | $99,355.09 |

#### Descriptive Statistics for Age

| Statistic | Value    |
|-----------|----------|
| Count     | 1000.0000|
| Mean      | 28.9090  |
| Std Dev   | 6.1721   |
| Min       | 23.0000  |
| 25%       | 25.0000  |
| 50% (Median) | 27.0000 |
| 75%       | 30.0000  |
| Max       | 60.0000  |



## 3. Feature Engineering
- Created the following additional columns for further exploration:
  - Total Compensation
  - Tenure
  - BIPOC
  - Bonus Percentage

#### Total Compensation 

```sh
# new column for total compensation
data["total comp"] = data["bonus"] + data["salary"]
data['total comp'].describe()
```

| Statistic | Value           |
|-----------|-----------------|
| Count     | 1000.000000     |
| Mean      | $95,137.39      |
| Std Dev   | $59,570.63      |
| Min       | $26,164.59      |
| 25%       | $60,782.60      |
| 50% (Median) | $78,779.94  |
| 75%       | $103,782.95     |
| Max       | $387,515.89     |

- The average total compensation is $95,137.39, with a wide range from $26,164.59 to $387,515.89, indicating significant variation in employee earnings.


#### Tenure (months)

```sh
# new column for tenure in months + find the descriptive statistics
today = pd.to_datetime(dt.date.today())
data['tenure (months)'] = (((today- data['startdate'])) /  np.timedelta64(1, 'D') / 30.417).astype('int64')
data['tenure (months)'].describe()
```


| Statistic | Value           |
|-----------|-----------------|
| Count     | 1000.000000     |
| Mean      | 52.319000|
| Std Dev   | 28.252353   |
| Min       | 5.000000   |
| 25%       | 27.000000    |
| 50% (Median) | 53.000000 |
| 75%       | 76.250000   |
| Max       | 101.000000  |

- Employees have a mean tenure of approximately 52 months, with the longest tenure being 101 months, showcasing a relatively moderate employee retention rate.

#### Bonus %

```sh
# new column for bonus % for every employee and descriptive stats
data["bonus %"] = (data['bonus']/data['salary'])*100
data["bonus %"].describe()
```


| Statistic | Value           |
|-----------|-----------------|
| Count     | 1000.000000     |
| Mean      | 9.807340  |
| Std Dev   | 6.078331  |
| Min       | 4.500000 |
| 25%       | 5.050000   |
| 50% (Median) | 9.400000 |
| 75%       | 10.800000  |
| Max       | 36.300000  |

- Average bonus percentage is approximately 9.8% of the salary, with extremes ranging from 4.5% to 36.3%, highlighting wide variety in bonus distribution.

#### BIPOC Column

```sh
# create BIPOC Column
data["bipoc"] = data["race"].str.contains('|'.join(['Asian', 'Hispanic','African American','Other']), regex=True)
data
```

- The BIPOC column is created by identifying individuals whose race is categorized as Asian, Hispanic, African American, or Other, highlighting diversity within the workforce.
- If an entry matches any of these categories listed in str.contains(), bipoc column  will returns True, otherwise it returns False.


## 4. Data Exploration by Groups


```sh
# Explore numeric data by groups
for i in data.select_dtypes(['category']):
    normed = pd.DataFrame(data[i].value_counts(normalize = False))  # counts for each category
    nonnormed = pd.DataFrame(data[i].value_counts(normalize = True))  # proportions for each category
    tenure = pd.DataFrame(data.groupby(i)['tenure (months)'].mean())  # average tenure by category
    totalcomp = pd.DataFrame(data.groupby(i)['total comp'].mean())  # average total compensation by category
    bonus = pd.DataFrame(data.groupby(i)['bonus %'].mean())  # average bonus percentage by category
    result = pd.concat([normed, nonnormed, tenure, totalcomp, bonus], axis=1)  # combine all results into a single dataframe
    print(result)
```

#### Descriptive Statistics by Gender

| Gender | Count | Proportion | Tenure (Months) | Total Comp | Bonus % |
|--------|-------|------------|-----------------|------------|---------|
| Male   | 510   | 0.51       | 52.53           | $97,504.68 | 10.05   |
| Female | 490   | 0.49       | 52.10           | $92,673.48 | 9.56    |



#### Descriptive Statistics by Race

| Race              | Count | Proportion | Tenure (Months) | Total Comp  | Bonus % |
|-------------------|-------|------------|-----------------|-------------|---------|
| Caucasian         | 543   | 0.543      | 52.51           | $97,807.10  | 10.06   |
| Asian             | 228   | 0.228      | 51.68           | $91,386.77  | 9.34    |
| Hispanic          | 96    | 0.096      | 54.14           | $90,118.37  | 9.61    |
| African American  | 90    | 0.090      | 53.07           | $91,691.82  | 9.50    |
| Other             | 43    | 0.043      | 47.63           | $99,728.60  | 10.16   |


#### Descriptive Statistics by Department

| Department               | Count | Proportion | Tenure (Months) | Total Comp  | Bonus % |
|--------------------------|-------|------------|-----------------|-------------|---------|
| Support                  | 96    | 0.096      | 52.52           | $84,615.79  | 9.60    |
| Product Management       | 94    | 0.094      | 50.59           | $109,373.59 | 10.52   |
| Sales                    | 94    | 0.094      | 54.83           | $73,675.43  | 8.98    |
| Services                 | 90    | 0.090      | 50.33           | $79,362.78  | 9.87    |
| Engineering              | 88    | 0.088      | 55.55           | $118,551.44 | 9.60    |
| Legal                    | 84    | 0.084      | 53.65           | $96,531.27  | 9.64    |
| Business Development     | 81    | 0.081      | 48.42           | $105,085.81 | 10.53   |
| Marketing                | 81    | 0.081      | 50.07           | $84,001.83  | 8.59    |
| Accounting               | 75    | 0.075      | 57.99           | $103,260.37 | 9.93    |
| Human Resources          | 73    | 0.073      | 51.10           | $93,772.50  | 10.57   |
| Training                 | 72    | 0.072      | 50.38           | $86,452.53  | 10.76   |
| Research and Development | 71    | 0.071      | 52.24           | $107,554.88 | 8.97    |
| Admin                    | 1     | 0.001      | 50.00           | $356,868.04 | 32.34   |



#### Descriptive Statistics by Employment Level

| Level       | Count | Proportion | Tenure (Months) | Total Comp   | Bonus % |
|-------------|-------|------------|-----------------|--------------|---------|
| Entry Level | 424   | 0.424      | 51.46           | $59,715.69   | 4.97    |
| Junior      | 360   | 0.360      | 52.63           | $84,859.78   | 10.03   |
| Mid-Level   | 126   | 0.126      | 53.69           | $134,350.39  | 14.93   |
| Senior      | 59    | 0.059      | 58.14           | $199,489.15  | 19.91   |
| VP          | 19    | 0.019      | 46.47           | $337,961.95  | 33.19   |
| Executive   | 11    | 0.011      | 38.45           | $344,751.45  | 33.72   |
| CEO         | 1     | 0.001      | 50.00           | $356,868.04  | 32.34   |



#### Descriptive Statistics by City

| City          | Count | Proportion | Tenure (Months) | Total Comp   | Bonus % |
|---------------|-------|------------|-----------------|--------------|---------|
| San Francisco | 391   | 0.391      | 53.04           | $90,686.99   | 9.44    |
| New York      | 246   | 0.246      | 50.69           | $96,212.92   | 9.85    |
| Chicago       | 191   | 0.191      | 49.50           | $105,016.77  | 10.61   |
| Boston        | 88    | 0.088      | 57.06           | $99,964.04   | 10.60   |
| Remote        | 84    | 0.084      | 55.19           | $85,182.90   | 8.70    |


From the tables above we see that:
- Males and females have nearly equal representation with little variation in tenure, but males earn slightly higher total compensation and bonuses on average.
- Engineering department has the highest average total compensation.
- As employees rise from entry to executive levels, there is a clear upward trend in both total compensation and bonus percentage, highlighting a strong correlation between seniority and earnings.
- Employees in Chicago have the highest average total compensation among the cities listed, while San Francisco, with the largest employee base, has the lowest average total compensation but a moderate bonus percentage.



```sh
# create histograms for each of the numeric variables
for x in data.select_dtypes(['int64','float64']):
   sns.displot(data[x], kde=False, height = 8)
```


<img width="570" alt="image" src="https://github.com/Hello-DataSkillUp/Jake_EDA1/assets/165890395/d24b1dc5-17b4-4ed1-8bbf-93390a903af2">

<img width="581" alt="image" src="https://github.com/Hello-DataSkillUp/Jake_EDA1/assets/165890395/f287715d-083b-4de7-956d-c99bf0f7c3ed">

<img width="612" alt="image" src="https://github.com/Hello-DataSkillUp/Jake_EDA1/assets/165890395/83112bc7-7c2f-4c2c-b09a-26451816b238">

<img width="619" alt="image" src="https://github.com/Hello-DataSkillUp/Jake_EDA1/assets/165890395/208be155-e02f-4e9e-92e9-a6054e4eb132">

<img width="624" alt="image" src="https://github.com/Hello-DataSkillUp/Jake_EDA1/assets/165890395/ad18ac6f-31e4-4eb1-ada1-b0458b4d2cac">

<img width="591" alt="image" src="https://github.com/Hello-DataSkillUp/Jake_EDA1/assets/165890395/e5c98b6e-395d-4321-a345-f5100f442db9">

<img width="596" alt="image" src="https://github.com/Hello-DataSkillUp/Jake_EDA1/assets/165890395/1d465e3e-ffd4-423a-9b7d-9e801d02b9a6">






## 5. Hypothesis Testing
### Compensation Equity
- No significance found.
#### All Employees and BIPOC Employees
```sh
#group employees by race and find descriptive statistics for total comp
data.groupby('race').agg({'total comp': ['mean', 'count',np.std]})
```
| Race             | Count | Mean     | Standard Deviation |
|------------------|-------|----------|--------------------|
| African American | 90    | 91691.82 | 51803.04           |  
| Asian            | 228   | 91386.77 | 51036.95           |
| Caucasian        | 543   | 97807.10 | 65198.58           |
| Hispanic         | 96    | 90118.37 | 48192.17           |
| Other            | 43    | 99728.60 | 65269.07           |
```sh
#create boxplot of total comp by race to look for any obvious outliers
data.boxplot(by = 'race', column = ['total comp'])
```
![comp boxplot by race](https://github.com/jakeqaverch/EDA/assets/170358772/1e17a25d-dd65-43d0-b05f-ef98694285a1)
```sh
bipoc = data[data['bipoc'] == True]
caucasian = data[data['race'] == 'Caucasian']
data.boxplot(by = 'bipoc', column = ['total comp'])
ttest_ind(bipoc['total comp'], caucasian['total comp'])
TtestResult(statistic=-1.545874354760165, pvalue=0.12245193827166366, df=998.0)
```
- While no significant outliers were found, I will compare groups individually
#### African American and Caucasian employees
```sh
#perform independent t-test of compensation between african american and caucasian employees
african_american = data[data['race'] == 'African American']
ttest_ind(african_american['total comp'], caucasian['total comp'])
TtestResult(statistic=-0.8464359312906196, pvalue=0.3976304858982638, df=631.0)
```
#### Hispanic and Caucasian employees
```sh
#perform independent t-test of compensation between hispanic and caucasian employees
hispanic = data[data['race'] == 'Hispanic']
ttest_ind(hispanic['total comp'], caucasian['total comp'])
TtestResult(statistic=-1.1030956875388647, pvalue=0.2704021710633812, df=637.0)
```
- This is surprising. While the means are close, the standard deviation is much higher which indicates a greater range. We will explore gender next.
#### Male and Female Employees
```sh
data.groupby('gender').agg({'total comp': ['mean', 'count',np.std]})
```
| Gender | Count | Mean     | Standard Deviation |
|--------|-------|----------|--------------------|
| Female | 490   | 92673.48 | 53437.63           |  
| Men    | 510   | 97504.68 | 64884.66           |
```sh
#create boxplot of above data
data.groupby('gender').agg({'total comp': ['mean', 'count',np.std]})
```
![image](https://github.com/jakeqaverch/EDA/assets/170358772/4c2540d3-1ec8-478c-ae24-43ae6a802ee9)

```sh
#perform independent t-test to determine statistical significance between male and female total compensation and create boxplot to visualize the data
male = data[data['gender'] == 'Male']
female = data[data['gender'] == 'Female']
ttest_ind(male['total comp'], female['total comp'])
TtestResult(statistic=1.282466173840882, pvalue=0.19997695595504292, df=998.0)
```
-As we saw with race, the standard deviation is higher indicating greater spread. But nothing significant here.

### BIPOC and Women Analysis
- Executive team breakdown by race and gender: Executives are mostly white males.
- Tenure by race and gender: No significance found.

### Executive Team Breakdown
#### Executive Team Breakdown by Race
```sh
#define executive group as anyone at the executive level or above and determine the racial breakdown of this group
execs = data[data['levelnumber'] >= 6]
pd.DataFrame(execs["race"].value_counts(normalize=True))
#visualize the above data
pl.pie(execs["race"].value_counts(),labels = ['Caucasian(92%)', 'African American (8%)', ' ', ' ',' '])
pl.show()
```
| Race             | Proportion |
|------------------|------------|
| African American | 0.08       |
| Asian            | 0          |
| Caucasian        | 0.92       |
| Hispanic         | 0          |
| Other            | 0          |

![image](https://github.com/jakeqaverch/EDA/assets/170358772/9c782405-df1a-45a0-a570-ead2707f8ab5)

-This is significant. Almost 92% of executives are white. We would need to investigate this further to deterimine the cause.

#### Executive Team Breakdown by Gender
```sh
#find gender breakdown of executives defined above
pd.DataFrame(execs["gender"].value_counts(normalize=True))
#visualize the above data
pl.pie(execs["gender"].value_counts(),labels = ['Male(92%)', 'Female (8%)'])
pl.show()
```
| Gender | Proportion |
|--------|------------|
| Female | 0.08       |
| Male   | 0.92       |

![image](https://github.com/jakeqaverch/EDA/assets/170358772/8628f797-7fe4-4a01-93df-e53eac98ad27)

- This is significant. Almost 92% of executives are male. We would need to investigate this further to deterimine the cause.

### Tenure (all values in months) by Gender and Race
#### Tenure by Gender
```sh
#create table of tenure by gender
data.groupby('gender').agg({'tenure (months)': ['mean', 'count',np.std]})
#visualize the above data
data.boxplot(by = 'gender', column = ['tenure (months)'])
```
| Gender | Count | Mean  | Standard Deviation |
|--------|-------|-------|--------------------|
| Female | 490   | 52.10 | 28.32              |  
| Men    | 510   | 52.53 | 28.22              |

![image](https://github.com/jakeqaverch/EDA/assets/170358772/ab5f2c90-153b-4f60-a2ae-108baf87829d)
```sh
#perform independent t-test to determine confirm lack of statistical significance between male and female tenure
ttest_ind(male['tenure (months)'], female['tenure (months)'])
TtestResult(statistic=0.2423963737638562, pvalue=0.8085228469796168, df=998.0)
```
- Nothing signficant was found here. Let's explore race next.
#### Tenure by Race
-First let's compare BIPOC and Caucasian employees
```sh
#create table of tenure by BIPOC status
data.groupby('bipoc').agg({'tenure (months)': ['mean', 'count',np.std]})
#visualize above data
data.boxplot(by = 'bipoc', column = ['tenure (months)'])
```
| Race      | Count | Mean  | Standard Deviation |
|-----------|-------|-------|--------------------|
| BIPOC     | 457   | 52.32 | 28.32              |  
| Non-BIPOC | 543   | 52.73 | 28.23              |

![image](https://github.com/jakeqaverch/EDA/assets/170358772/e0e60edb-77b6-400a-9b88-e4e6077cb2a0)

```sh
#perform independent t-test of BIPOC and non-BIPOC employee tenure
ttest_ind(caucasian['tenure (months)'], bipoc['tenure (months)'])
TtestResult(statistic=0.22813352262198297, pvalue=0.8195891719210135, df=998.0)
```
- Nothing significant here. Next let's explore groups individually

```sh
#create table of tenure by gender
data.groupby('race').agg({'tenure (months)': ['mean', 'count',np.std]})
#visualize the above data
data.boxplot(by = 'race', column = ['tenure (months)'])
```
| Race             | Count | Mean  | Standard Deviation |
|------------------|-------|-------|--------------------|
| African American | 90    | 53.07 | 26.87              |  
| Asian            | 228   | 51.68 | 28.50              |
| Caucasian        | 543   | 52.51 | 28.22              |
| Hispanic         | 96    | 54.14 | 28.82              |
| Other            | 43    | 47.63 | 29.51              |

![image](https://github.com/jakeqaverch/EDA/assets/170358772/fd619f39-12a3-47f7-b9c8-d59610d2c1d3)

-Nothing signfiicant here.

### Additional Group Analysis
- Department-wise analysis. SHOULD THIS BE HERE
- Tenure and tenure by level.
- Compensation (bonus %) analysis.
- Junior Employees: Junior salespeople have above-average tenure and below-average compensation.
- Location-based analysis: High number of junior employees in SFO, fewer in Boston.

####Junior Employees
```sh
#define Junior employees as levels 2 and below and determine which department they sit in
data["Junior"] = data['levelnumber']<=2
juniornormed = data.groupby('department')['Junior'].value_counts(normalize = True)
juniornonnormed = data.groupby('department')['Junior'].value_counts(normalize = False)
result = pd.concat([juniornonnormed, juniornormed], axis=1)  # combine all results into a single dataframe
print(result)
```
| Department               | Count | Proportion |
|--------------------------|-------|------------|
| Accounting               | 56    | 0.75       |
| Admin                    | 0     | 0          |
| Business Developmen      | 58    | 0.72       |
| Human Resources          | 62    | 0.81       |
| Legal                    | 70    | 0.74       |
| Marketing                | 70    | 0.86       |
| Product Management       | 72    | 0.77       |
| Research and Development | 61    | 0.87       |
| Sales                    | 78    | 0.83       |
| Services                 | 71    | 0.79       |
| Support                  | 77    | 0.80       |
| Training                 | 51    | 0.71       |
| Admin                    | 0     | 0.75       |

-The Sales dept has a high number of junior employees.
-R+D and Marketing have a high proportion of junior employees.
-Let's explore this further and explore tenure

##### Junior Employee Tenure
-Mean Junior employee tenure is 52.30
```sh
#visualize tenure by department as a whole to see where recent hiring may have been
data.boxplot(by = 'department', column = ['tenure (months)'], figsize = (7, 5), rot = 80, grid = False)
```
![image](https://github.com/jakeqaverch/EDA/assets/170358772/89079658-0096-47d9-8fa6-487b5e67d993)

-Tenure in BD and Services is lower than average
-Tenure in Accounting, Legal Sales, and Support is relatviely high
-Let's look at Junior employees specifically
```sh
#determine avg tenure by department for junior employees
junior_employees.groupby('department')['tenure (months)'].mean()
```
| Department               | Avg Tenure |
|--------------------------|------------|
| Accounting               | 58.21      |
| Business Developmen      | 56.03      |
| Engineering              | 56.03      |
| Human Resources          | 50.41      |
| Legal                    | 53.79      |
| Marketing                | 49.53      |
| Product Management       | 51.42      |
| Research and Development | 52.51      |
| Sales                    | 55.46      |
| Services                 | 50.90      |
| Support                  | 49.95      |
| Training                 | 52.37      |

-The Sales, Accounting and Engineering team have higher junior tenure next let's explore comepnsation next which is often determined by job type and location
##### Junior Employee Tenure
-Mean Junior Employee total compensation is $71261.45
```sh
#determine avg total comp by department for junior employees
junior_employees.groupby(['department','city']).agg({"total comp" : ["count", "mean"]})
```
| Department               | City          | Count | Mean     |
|--------------------------|---------------|-------|----------|
| Accounting               | Boston        | 7     | 78504.60 |
|                          | Chicago       | 5     | 67324.10 |
|                          | New York      | 22    | 70429.68 |
|                          | Remote        | 4     | 70662.11 |
|                          | San Fransisco | 18    | 81669.47 |
| Business Development     | Boston        | 7     | 76190.16 |
|                          | Chicago       | 12    | 76257.37 |
|                          | New York      | 10    | 74334.26 |
|                          | Remote        | 6     | 72068.75 |
|                          | San Fransisco | 23    | 73389.10 |
| Engineering              | Boston        | 4     | 97521.06 |
|                          | Chicago       | 12    | 95737.22 |
|                          | New York      | 23    | 97652.15 |
|                          | Remote        | 5     | 98806.18 |
|                          | San Fransisco | 25    | 96256.20 |
| Human Resources          | Boston        | 4     | 76770.34 |
|                          | Chicago       | 5     | 55960.07 |
|                          | New York      | 14    | 67199.11 |
|                          | Remote        | 5     | 60029.50 |
|                          | San Fransisco | 31    | 60553.23 |
| Legal                    | Boston        | 3     | 62566.75 |
|                          | Chicago       | 10    | 73129.43 |
|                          | New York      | 18    | 78834.04 |
|                          | Remote        | 5     | 68450.39 |
|                          | San Fransisco | 26    | 72286.72 |
| Marketing                | Boston        | 6     | 67215.99 |
|                          | Chicago       | 16    | 68532.34 |
|                          | New York      | 12    | 72848.87 |
|                          | Remote        | 6     | 72465.50 |
|                          | San Fransisco | 30    | 71299.12 |
| Product Management       | Boston        | 3     | 62445.33 |
|                          | Chicago       | 20    | 84983.96 |
|                          | New York      | 17    | 88662.47 |
|                          | Remote        | 6     | 81515.82 |
|                          | San Fransisco | 26    | 74914.83 |
| Research and Development	| Boston        | 4     | 94268.20 |
|                          | Chicago       | 16    | 85578.17 |
|                          | New York      | 14    | 94030.18 |
|                          | Remote        | 8     | 89654.37 |
|                          | San Fransisco | 19    | 91147.32 |
| Sales                   	| Boston        | 6     | 56983.83 |
|                          | Chicago       | 12    | 51642.33 |
|                          | New York      | 17    | 51780.62 |
|                          | Remote        | 7     | 55442.74 |
|                          | San Fransisco | 36    | 54468.03 |
| Services                	| Boston        | 7     | 49011.55 |
|                          | Chicago       | 15    | 56778.19 |
|                          | New York      | 13    | 59434.48 |
|                          | Remote        | 6     | 65383.22 |
|                          | San Fransisco | 30    | 59717.50 |
| Support                 	| Boston        | 5     | 78556.00 |
|                          | Chicago       | 15    | 69552.34 |
|                          | New York      | 23    | 61522.39 |
|                          | Remote        | 9     | 69508.79 |
|                          | San Fransisco | 25    | 65067.09 |
| Training                	| Boston        | 6     | 51584.08 |
|                          | Chicago       | 7     | 51775.36 |
|                          | New York      | 12    | 58051.08 |
|                          | Remote        | 5     | 52036.07 |
|                          | San Fransisco | 21    | 50175.10 |

-Junior sales people have above average tenure and below average compensation. We need more data to explore further but it is interesting.
-A bulk of entry level talent has been hired in SF which is a very expensive market. This should be examined further.
-There are very few remote junior employees across the organization.

#### Tenure by Level
```sh
#boxplot of tenure by level
data.boxplot(by = 'levelnumber', column = 'tenure (months)', figsize = (5, 5))
```
![image](https://github.com/jakeqaverch/EDA/assets/170358772/d72d1419-8024-4c19-abae-21599ddbb2ae)

- Tenure at the executive level is relatviely low. There could be many reaons for this so it would require more data to explore further. It could be an issue of small sample size.

#### Compensation Analysis: Bonus %
```sh
#boxplot of tenure by level
data.boxplot(by = 'levelnumber', column = 'tenure (months)', figsize = (5, 5))
```
| Department               | Level        | Bonus % Mean | Bonus % Std  |
|--------------------------|--------------|--------------|--------------|
| Accounting               | Entry Level	 | 33           |	4.97         |                 
|                          | Junior       |	23	          | 10.04        |
|                          | Mid-Level	   | 12	          | 14.71        |
|                          | Senior       |	4	           | 19.60        | 
|                          | VP	          | 2            |	31.52        |
|                          | Executive    |	1	           | 32.01        |
| Admin                    |	CEO	         | 1	           | 32.34        |
| Business Development	    | Entry Level	 | 33           |	4.97         |
|                          | Junior       |	25	          | 10.08        |
|                          | Mid-Level	   | 15	          | 14.93        |
|                          | Senior	      | 4	           | 20.2         |
|                          | VP           |	3	           | 32.12        |
|                          | Executive	   | 1	           | 35.97        |
| Engineering	             | Entry Level	 | 36	          | 4.91         |
|                          | Junior	      | 33	          | 9.97         |
|                          | Mid-Level	   | 14	          | 15.09        |
|                          | Senior	      | 3	           | 19.53        |
|                          | VP	          | 1            |	34.98        |
|                          | Executive    |	1	           | 34.32        |
| Human Resources          |	Entry Level  |	29	          | 4.88         |
|                          | Junior	      | 30           |	10.00        |
|                          | Mid-Level    |	4	           | 14.51        |
|                          | Senior	      | 5            |	21.16        | 
|                          | VP           |	4	           | 32.84        |
|                          | Executive	   | 1            |	34.65        |
| Legal                    | Entry Level	 | 35	          | 5.00         |
|                          | Junior       |	27	          | 10.04        |
|                          | Mid-Level    |	14	          | 14.75        |
|                          | Senior	      | 8	           | 19.68        |
| Marketing                | Entry Level	 | 40           |	5.018        |
|                          | Junior	      | 30	          | 10.02        |
|                          | Mid-Level	   | 7	           | 14.74        |
|                          | Senior	      | 3	           | 19.53        | 
|                          | Executive	   | 1	           | 32.67        |
| Product Management       | Entry Level	 | 34           |	4.92         |
|                          | Junior       |	38           |	10.01        | 
|                          | Mid-Level    |	9	           | 15.02        |
|                          | Senior       |	10	          | 20.02        |
|                          | VP           |	2	           | 35.31        |
|                          | Executive    |	1	           | 35.97        |
| Research and Development | Entry Level  |	35	          | 5.07         |
|                          | Junior       |	26           | 9.96         |
|                          | Mid-Level	   | 5	           | 14.79        |
|                          | Senior	      | 3	           | 19.80        |
|                          | VP           |	1	           | 34.98        |
|                          | Executive    |	1            |	32.01        |
| Sales                    | Entry Level	 | 48           |	4.95         |
|                          | Executive	   | 1	           | 36.3         |
|                          | Junior       |	30	          | 9.92         | 
|                          | Mid-Level    |	9	           | 14.37        |
|                          | Senior       | 4	           | 19.20        |
|                          | VP	          | 2	           | 33.50        |
Services	                  | Entry Level  |	33           |	4.95         |
|                          | Junior      	| 38           |	10.05        |
|                          | Mid-Level	   | 11	          | 15.20        |
|                          | Senior      	| 7            | 20.03        |
|                          | Executive   	| 1	           | 34.65        |
|Support                  	| Entry Level  | 38           | 5.051        |
|                          | Junior	      | 39           |	10.05        |
|                          | Mid-Level    |	14	          | 15.32        |
|                          | Senior       | 3           	| 19.73        |
|                          | VP           |	1	           | 33.33        |
|                          | Executive    |	1	           | 30.69        |
|Training                  | Entry Level  |	30          	| 4.98         |
|                          | Junior	      | 21           | 10.24        |
|                          | Mid-Level    | 12	          | 15.03        |
|                          | Senior	      | 5	           | 19.80        |
|                          | VP	          |3             |	33.00        |
|                          | Executive	   | 1	           | 31.68        |

- Bonus % does not vary much by level and department nothing significant here.

#### Location Based Analysis
##### Overall Location Data
```sh
#breakdown of numeric variables by city
citylvl = data.groupby("city").agg({"levelnumber" : ["count", "mean",np.std]}) #find city level data
citytenure = data.groupby("city").agg({"tenure (months)" : ["mean", np.std]}) #find city tenure data
citycomp = data.groupby("city").agg({"total comp" : ["mean", np.std]}) #find comp level data
cityage = data.groupby("city").agg({"age" : ["mean", np.std]}) #find age level data
citytable = pd.concat([citylvl,citytenure,citycomp,cityage], axis=1) #combine data
citytable #print data
```
| City          | Count | Level (Mean) | Level (Std) | Tenure (Mean) | Tenure (Std) | Total Comp (Mean) | Total Comp (Std) | Age (Mean) | Age (Std) |
|---------------|-------|--------------|-------------|---------------|--------------|-------------------|------------------|------------|-----------|
| Boston        | 88    | 2.07         | 1.08        | 57.06         | 27.48        | 99964.04          | 59445.33         | 30.27      | 6.96      |
| Chicago       | 191   | 2.07	        | 1.24        |	49.50         |	28.45        | 105016.77         |	74807.67         |	29.52      | 7.42      |
| New York      | 246   | 1.95	        | 1.14        |	50.69         | 28.10        |	96212.92          |	60034.90         |	28.60      |	5.61      |
| Remote        | 84    | 1.74	        | 0.89        |	55.19         |	28.18        |	85182.90          | 43679.38         | 28.02      |	5.71      |
| San Francisco | 391   | 1.87	        | 0.98        | 53.04         |	28.36        |	90687.00          |	53011.22         |	28.69      | 5.69      |

- It looks like Boston has higher mean level number with low varience let's explore this further below.
- Remote compesnation is conserderably lower with  a lower varience which is not surpiresing.
- Chicago compensation is higher with higher than average varience which should be investigated further.

##### Locaiton Level Data Expanded
```sh
#deeper dive into level and location data
citylevelnonrm = data.groupby('city')['level'].value_counts(normalize=True) #normalized city location data
citylevelnonnrm = data.groupby('city')['level'].value_counts(normalize=False) #non-normalized city location data
execloc = pd.concat([citylevelnonnrm,citylevelnonrm], axis=1) #combine data
execloc #print data
```
| City          | Level       | Count | %     | 
|---------------|-------------|-------|-------|
| Boston	       | Entry Level	| 33	   | 0.38  |
|               | Junior	     | 29	   | 0.33  |
|               | Mid-Level	  | 15	   | 0.17  |
|               | Senior	     | 9     |	0.10  |
|               | VP	         | 2	    | 0.023 |
| Chicago	      | Entry Level	| 81    | 0.42  |
|               | Junior	     | 64    |	0.34  |
|               | Mid-Level	  | 20	   | 0.10  |
|               | Senior	     | 14	   | 0.073 |
|               | VP	         | 8	    | 0.042 |
|               | Executive	  | 4     |	0.02  |
| New York      |	Entry Level	| 102	  | 0.41  |
|               | Junior	     | 93    |	0.38  |
|               | Mid-Level   |	28	   | 0.11  |
|               | Senior	     | 15    |	0.06  |
|               | Executive	  | 4	    | 0.016 |
|               | VP          |	3	    | 0.012 |
|               | CEO	        | 1	    | 0.004 |
| Remote	       | Entry Level |	39    |	0.46  |
|               | Junior	     | 33    |	0.39  |
|               | Mid-Level	  | 9     |	0.11  |
|               | Senior	     | 2     |	0.024 | 
|               | Executive	  | 1     |	0.012 |
| San Francisco | Entry Level	| 169   |	0.43  |
|               | Junior	     | 141   |	0.36  |
|               | Mid-Level	  | 54	   | 0.14  |
|               | Senior	     | 19	   | 0.049 |
|               | VP	         | 6     |	0.015 |
|               | Executive   |	2     |	0.005 |

 - There are not as many junior employees in Boston.
 - There is a high number of junior employees in SFO which is an expensive market to hire into.

### Further Exploration
- Tenure by level: Indicates lower tenure at the executive level, requiring more data to understand why.
- Bonus percentage by department and level: No significant outliers found.

## Detailed Analysis

### Hypothesis: Compensation Equity
- **Initial Finding:** No significant difference found in compensation equity.
- **Detailed Steps:**
  - Explore differences by BIPOC status and gender.
  - Analyze executive team breakdown by race and gender.
  - Investigate tenure by race and gender.

### BIPOC and Gender Analysis
- **Key Insights:**
  - **Executive Team:** Predominantly white males.
  - **Tenure:** No significant difference found across race and gender.
  - **Junior Employees:** Higher tenure but lower compensation, especially in sales.

### Department and Location Analysis
- **Departments:**
  - Sales, Accounting, and Engineering have higher junior tenure.
  - Junior salespeople have above-average tenure and below-average compensation.
- **Locations:**
  - Boston has fewer junior employees with a higher mean level number.
  - SFO has a high number of junior employees, an expensive market for hiring.

### Further Exploration
- **Tenure by Level:** Indicates lower tenure at the executive level.
- **Bonus Percentage:** No significant variance by department and level.

## Conclusion
The analysis highlights key areas for further investigation, particularly around compensation equity, executive team diversity, and tenure trends by department and location.

## Future Work
- Investigate the reasons behind lower tenure at the executive level.
- Explore additional datasets to understand compensation and tenure dynamics better.
- Conduct more detailed analysis on bonus percentages to identify potential outliers.
