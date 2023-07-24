import pandas as pd

# Load the Titanic dataset
titanic = pd.read_csv('titanic.csv')

# Create a new feature for family size
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

# Create a new feature for whether the passenger was alone or not
titanic['IsAlone'] = 0
titanic.loc[titanic['FamilySize'] == 1, 'IsAlone'] = 1

# Create a new feature for the title of the passenger
titanic['Title'] = titanic['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

# Print the first few rows of the modified dataset
print(titanic.head())

titanic['survives']=titanic.apply(lambda row: 1 if row['IsAlone']==0 else 0,axis=1)