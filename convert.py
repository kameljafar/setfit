import pandas as pd
import json

# Load the JSON data
with open('C:\\allProjects\FATWA_PROJECT\\fatwa_classification\data\documents_with_annotations_2_11_2024.json', 'r') as json_file:
    data = json.load(json_file)

# Extract relevant information from the JSON data
rows = []
for item in data:

    row = {
        'text': item['text'],
        'first_cates': item['law_type'],
        'sec_cates': item['dominant_justification'],
        'cates': f"{item['law_type']},{item['dominant_justification']}"
    }
    rows.append(row)

# Create a DataFrame
df = pd.DataFrame(rows)

# Save the DataFrame to Excel
df.to_excel('documents_with_annotations_2_11_2024.xlsx', index=False)
