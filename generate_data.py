import os
os.makedirs('data', exist_ok=True)
import pandas as pd
import numpy as np

np.random.seed(42)
n = 600

data = pd.DataFrame({
    'screen_time_hrs':   np.round(np.random.uniform(1, 12, n), 2),
    'unlocks_per_day':   np.random.randint(10, 200, n),
    'night_usage':       np.random.randint(0, 2, n),
    'social_media_hrs':  np.round(np.random.uniform(0, 6, n), 2),
    'app_switches_hr':   np.random.randint(5, 80, n),
    'notifications_day': np.random.randint(10, 300, n),
    'sleep_hours':       np.round(np.random.uniform(3, 9, n), 2),
    'age':               np.random.randint(13, 60, n),
})

def label(row):
    score = 0
    if row['screen_time_hrs'] > 7:    score += 2
    if row['screen_time_hrs'] > 10:   score += 1
    if row['unlocks_per_day'] > 100:  score += 2
    if row['unlocks_per_day'] > 150:  score += 1
    if row['night_usage'] == 1:       score += 1
    if row['social_media_hrs'] > 3:   score += 2
    if row['social_media_hrs'] > 5:   score += 1
    if row['sleep_hours'] < 6:        score += 2
    if row['sleep_hours'] < 5:        score += 1
    if row['notifications_day'] > 150: score += 1
    if row['notifications_day'] > 220: score += 1
    if row['app_switches_hr'] > 50:   score += 1

    if score <= 3:  return 'Low'
    elif score <= 6: return 'Medium'
    else:           return 'High'

data['risk_level'] = data.apply(label, axis=1)
data.to_csv('data/phone_usage.csv', index=False)
print("✅ Dataset generated: data/phone_usage.csv")
print(data['risk_level'].value_counts())
