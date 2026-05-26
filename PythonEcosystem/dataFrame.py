import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Salary': [50000, 60000, 70000, 80000],
    'Department': ['HR', 'IT', 'Finance', 'Marketing']
}

df = pd.DataFrame(data)

# Select a column
print(df['Age'])

# Filter rows
print(df[df['Salary'] > 60000])

# Group by department and calculate average salary
print(df.groupby('Department')['Salary'].mean())