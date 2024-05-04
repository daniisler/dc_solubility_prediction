import numpy as np
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(_file_))
input_directory = os.path.join(PROJECT_ROOT, 'cure')
output_directory = os.path.join(PROJECT_ROOT, 'finalclean/')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        #Read each CSV file
        file_path = os.path.join(input_directory, filename)
        data = pd.read_csv(file_path)

        #Sort data by 'smiles' and 'weight' in descending order (higher weight is better)
        sorted_data = data.sort_values(by=['smiles', 'weight'], ascending=[True,False])

        #Remove duplicates , keeping the first entry after sorting, which is the one with higher weight
        best_values_data = sorted_data.drop_duplicates(subset='smiles', keep='first')

        #Sort the cleaned data to the output directory with the same filename
        output_file_path = os.path.join(output_directory, filename)
        best_values_data.to_csv(output_file_path, index=False)

print('All done :)')

print('Cured data (before cleaning):')
path = input_directory
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        full_path = os.path.join(path, filename)  # Create the full path to the file
        with open(full_path, 'r', encoding="latin-1") as fileObj:
            # -1 to exclude the header
            print("Rows Counted {} in the csv {}:".format(len(fileObj.readlines()) - 1, filename))

print('Cleaned data (after cleaning):')
path = output_directory
for filename in os.listdir(path):
    full_path = os.path.join(path, filename)  # Create the full path to the file
    with open(full_path, 'r', encoding="latin-1") as fileObj:
        # -1 to exclude the header
        print("Rows Counted {} in the csv {}:".format(len(fileObj.readlines()) - 1, filename))


#join all csv files
# List to hold dataframes
df_list = []

for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        #for all csv files in \cure, append them to a list
        full_path = os.path.join(input_directory, filename)
        df = pd.read_csv(full_path, index_col=None, header=0)
        df_list.append(df)

        # Concatenate all data into one DataFrame
        combined_csv = pd.concat(df_list, axis=0, ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
combined_csv.to_csv("combined_data.csv", index=False)
print('I combined all the \cure data into one file :))')

#Read combined CSV file
filename = 'combined_data.csv'
data = pd.read_csv(filename)

#Sort data by 'smiles' and 'weight' in descending order (higher weight is better)
sorted_data = data.sort_values(by=['smiles', 'weight'], ascending=[True,False])

#Remove duplicates , keeping the first entry after sorting, which is the one with higher weight
best_values_data = sorted_data.drop_duplicates(subset='smiles', keep='first')

#Sort the cleaned data to the output directory with the same filename
best_values_data.to_csv("combined_cleaned_data.csv", index=False)
print('Finiiiished :)')



print('Cured data (before cleaning):')
filename = 'combined_data.csv'

with open(filename, 'r', encoding="latin-1") as fileObj:
    lines = fileObj.readlines()  # Read once and use multiple times
    a = len(lines) - 1  # -1 to exclude the header
    print("Rows Counted {} in the csv {}:".format(a, filename))

print('Cleaned data (after cleaning):')

filename = 'combined_cleaned_data.csv'

with open(filename, 'r', encoding="latin-1") as fileObj:
    lines = fileObj.readlines()  # Read once and use multiple times
    b = len(lines) - 1  # -1 to exclude the header
    print("Rows Counted {} in the csv {}:".format(b, filename))

print(a - b)