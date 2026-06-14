import pandas as pd

# ==============================================================
print("...1...")
# ==============================================================
df = pd.read_csv('physics_solar_panel_lab_dataset.csv')
print(df.head())
df.info()

radky, sloupce = df.shape 
print(f"Dataset má {radky} řádků a {sloupce} sloupců.")

# ==============================================================
print("...2...")
# ==============================================================
# Převod číselných sloupců a ošetření desetinných čárek
numeric_cols = ['lamp_distance_cm', 'angle_deg', 'light_intensity_lux', 'temperature_c', 'voltage_v', 'current_a', 'power_w']
for col in numeric_cols:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(',', '.')
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Převod data
df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str).str.replace('.', '-'), errors='coerce')

# Filtrace nevalidních hodnot
df = df[
    (df['lamp_distance_cm'] >= 0) & 
    (df['light_intensity_lux'] >= 0) & 
    (df['current_a'] >= 0) & 
    (df['power_w'] >= 0) & 
    (df['angle_deg'] <= 90)
]
df = df.dropna(subset=['timestamp'])

# Výplň chybějících hodnot
df['room'] = df['room'].fillna('Unknown').astype(str).str.strip().str.capitalize()
df['operator'] = df['operator'].fillna('Unknown').astype(str).str.strip()
df['notes'] = df['notes'].fillna('No notes')

# Odstranění duplicit a úprava textových sloupců
df = df.drop_duplicates(subset=['measurement_id'], keep='first')

text_cols = ['team', 'panel_id', 'weather']
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()

# Mapování textových hodnot
weather_mapping = {'suny': 'sunny', 'indoor lamp': 'indoor-lamp'}
df['weather'] = df['weather'].replace(weather_mapping)
df['panel_id'] = df['panel_id'].str.replace('_', '-')

print("Čištění dokončeno. Aktuální počet řádků:", len(df))

# ==============================================================
print("...3...")
# ==============================================================
df['power_calc'] = df['voltage_v'] * df['current_a']
print(df[['measurement_id', 'power_w', 'power_calc']].head())

max_rozdil = (df['power_w'] - df['power_calc']).abs().max()
print(f"Maximální rozdíl v datech je: {max_rozdil} W")

# ==============================================================
print("...4...")
# ==============================================================
uhly_analyza = df.groupby('angle_deg')['power_calc'].mean().reset_index()
uhly_analyza.columns = ['Úhel (°)', 'Průměrný výkon (W)']
uhly_analyza = uhly_analyza.sort_values(by='Úhel (°)')
print(uhly_analyza.to_string(index=False))

# ==============================================================
print("...5...")
# ==============================================================
df_clean = df.dropna(subset=['light_intensity_lux', 'power_calc'])
korelace = df_clean['light_intensity_lux'].corr(df_clean['power_calc'])
print(f"Korelační koeficient: {korelace:.4f}")

# ==============================================================
print("...6...")
# ==============================================================

# Analýza podle počasí
analyza_prostredi = df.groupby('weather')['power_calc'].mean().reset_index()
analyza_prostredi.columns = ['Podmínky (Počasí)', 'Průměrný výkon (W)']

# Analýza podle umístění (vnitřní / vnější)
df['room_type'] = df['room'].apply(lambda x: 'Outdoor (Střecha)' if 'Roof' in str(x) else 'Indoor (Laboratoř)')
analyza_mistnosti = df.groupby('room_type')['power_calc'].mean().reset_index()
analyza_mistnosti.columns = ['Prostředí', 'Průměrný výkon (W)']

print("Srovnání podle počasí:")
print(analyza_prostredi.to_string(index=False))
print("\nSrovnání podle umístění:")
print(analyza_mistnosti.to_string(index=False))

# ==============================================================
print("...7...")
# ==============================================================
# ...

# ==============================================================
print("...8...")
# ==============================================================
# ...
