
def age_group(age):
    if age < 30:
        return 'young'
    elif age < 50:
        return 'middle'
    else:
        return 'senior'

data = {
    'City':  ['Alger', 'Annaba', 'Oran', 'Alger', 'Oran', 'Annaba', 'Alger', 'Oran'],
    'Department': ['Cardiology', 'Neurology', 'Orthopedics', 'Cardiology', 'Neurology', 'Orthopedics', 'Cardiology', 'Neurology'],
    'Age': [45, 60, 30, 50, 40, 70, 35, 55],
    'DaysAdmitted': [5, 8, 3, 7, 4, 6, 2, 9],
    'DailyCost': [200, 300, 150, 220, 280, 160, 210, 290],
    'Satisfaction': [4.5, 3.8, 4.2, 4.0, 4.1, 3.5, 4.7, 3.9],
    'Readmitted': [0, 1, 0, 0, 1, 1, 0, 1],
}

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(data)

df['TotalCost'] = df['DaysAdmitted'] * df['DailyCost']
df['AgeGroup'] = df['Age'].apply(age_group)
df['Bonus'] = df['DailyCost'] * 0.1
df['RiskScore'] = df['DaysAdmitted'] * 0.4 + df['Age'] * 0.3 + (100 - df['Satisfaction'] * 10) * 0.3

sns.barplot(df, x='City', y='Satisfaction', hue='Department')
plt.show()

sns.barplot(df, x='City', y='Satisfaction')
plt.show()

sns.boxplot(data=df, x='City', y='RiskScore')
plt.show()

sns.violinplot(data=df, x='City', y='RiskScore')
plt.show()

correlation = df[['Age', 'DaysAdmitted', 'DailyCost', 'Satisfaction', 'Readmitted', 'TotalCost', 'Bonus', 'RiskScore']].corr()
sns.heatmap(correlation)
plt.show()
