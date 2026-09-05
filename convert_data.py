import json
import pandas as pd

# 1. Process Crop Soil Profiles
df_crop = pd.read_excel('data/Crop_Predication_dataset.xlsx')
crop_profiles = {}

for crop, group in df_crop.groupby('label'):
    crop_profiles[str(crop).lower()] = {
        "n_optimal": [round(float(group['N'].min()), 1), round(float(group['N'].max()), 1)],
        "p_optimal": [round(float(group['P'].min()), 1), round(float(group['P'].max()), 1)],
        "k_optimal": [round(float(group['K'].min()), 1), round(float(group['K'].max()), 1)],
        "ph_range": [round(float(group['ph'].min()), 1), round(float(group['ph'].max()), 1)]
    }

with open('server/crop_profiles.json', 'w') as f:
    json.dump(crop_profiles, f, indent=2)

# 2. Process District Soil Profiles (reads CSV)
df_soil = pd.read_csv('data/Soil data.csv')
district_profiles = {}

for district, group in df_soil.groupby('District'):
    district_profiles[str(district).strip()] = {
        "avg_n": round(float(group['Nitrogen Value'].mean()), 2),
        "avg_p": round(float(group['Phosphorous value'].mean()), 2),
        "avg_k": round(float(group['Potassium value'].mean()), 2),
        "avg_ph": round(float(group['pH'].mean()), 2)
    }

with open('server/district_soil.json', 'w') as f:
    json.dump(district_profiles, f, indent=2)

print("Generated server/crop_profiles.json and server/district_soil.json successfully.")
