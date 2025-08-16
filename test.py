# from tabulate import tabulate

# # The restructured data as a dictionary of lists of dictionaries
# models = {
#     'Falcon': [
#         {'version': '1.0', 'distance': '200.50', 'speed': '62.66', 'stability': '8.50'},
#         {'version': '1.1', 'distance': '220.75', 'speed': '71.21', 'stability': '9.00'}
#     ],
#     'Skyhawk': [
#         {'version': '1.0', 'distance': '180.25', 'speed': '55.50', 'stability': '7.80'}
#     ]
# }

# def create_combined_table(data):
#     """
#     Generates a single ASCII table with model names as internal headers.
#     """
#     all_data = []
    
#     # Get the column headers from the first entry to ensure consistency
#     headers = list(data['Falcon'][0].keys())

#     for model_name, versions in data.items():
#         # Add the model name as a list with a blank for each column
#         # This will be used as a header row in the table
#         all_data.append([model_name] + [''] * (len(headers) - 1))
        
#         # Add the column headers for the version data
#         all_data.append(headers)
        
#         # Add the data rows
#         for version in versions:
#             all_data.append(list(version.values()))

#     # Generate the table with an empty list for headers to avoid the TypeError
#     return tabulate(all_data, headers=[], tablefmt="grid")

# # Print the final output
# print(create_combined_table(models))


import csv
import os

from lucidraft import R


def view_models():
    outb = 'outputs'
    models = {}

    if not os.path.exists(outb):
        print(R + "❌ No models found! Create some planes first! ✈️")
        return

    for model_name in os.listdir(outb):
        model_path = f'{outb}/{model_name}'

        if os.path.isdir(model_path):
            for version in os.listdir(model_path):

                # version = str(version)
                amp = f'{model_path}/{version}/avg_metrics.csv'
                with open(amp, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        distance = str(row['Distance_px'])
                        speed = str(row['Speed_px_per_s'])
                        stability = str(row['Stability'])
  
                        models[model_name].append({
                            'version': str(version),
                            'Distance': distance,
                            'Speed': speed,
                            'Stability': stability
                        })

    return models
print(view_models())