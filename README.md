
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

### BIPOC and Women Analysis
- Executive team breakdown by race and gender: Executives are mostly white males.
- Tenure by race and gender: No significance found.
- Junior Employees: Junior salespeople have above-average tenure and below-average compensation.

### Additional Group Analysis
- Department-wise analysis.
- Tenure and tenure by department.
- Compensation analysis.
- Location-based analysis: High number of junior employees in SFO, fewer in Boston.

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
