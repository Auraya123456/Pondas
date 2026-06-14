import polars as pl


# ==============================================================================
print("...1...")
# ==============================================================================
print("1. NAČTENÍ DATASETU A ZOBRAZENÍ ZÁKLADNÍCH INFORMACÍ")


df = pl.read_csv(
   "physics_solar_panel_lab_dataset.csv",
   schema_overrides={
       "voltage_v": pl.String,
       "current_a": pl.String,
       "temperature_c": pl.String,
       "lamp_distance_cm": pl.String
   }
)


print(f"Původní rozměry datasetu (shape): {df.shape}")
print("\nPrvních 5 řádků surových dat:")
print(df.head(5))




# ==============================================================================
print("...2...")
# ==============================================================================
print("\n2. ČIŠTĚNÍ DAT, PŘEVODY TYPŮ A ODSTRANĚNÍ TEXTOVÝCH JEDNOTEK Z ČÍSEL")


# Robustní vyčištění textových anomálií a jednotek
df_clean = df.with_columns([
   pl.col("room").str.strip_chars(),
   pl.col("operator").str.strip_chars(),
   pl.col("weather").str.strip_chars(),
])


df_clean = df_clean.with_columns([
   # Sjednocení textu počasí
   pl.col("weather")
     .str.replace("suny", "sunny")
     .str.replace("indoor lamp", "indoor-lamp"),
  
   # OPRAVA: Přidán strict=False. Pokud formát neodpovídá, vloží se null místo pádu programu.
   pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
  
   # Odstranění jednotek 'V', oprava čárek a převod na Float64
   pl.col("voltage_v")
     .str.replace_all(r"[^\d,\.]", "")
     .str.replace(",", ".")
     .str.strip_chars()
     .cast(pl.Float64),
  
   # Odstranění jednotek 'A', oprava čárek a převod na Float64
   pl.col("current_a")
     .str.replace_all(r"[^\d,\.]", "")
     .str.replace(",", ".")
     .str.strip_chars()
     .cast(pl.Float64),
  
   # Odstranění textu 'C' z teploty
   pl.col("temperature_c")
     .str.replace_all(r"[^\d,\.]", "")
     .str.replace(",", ".")
     .str.strip_chars()
     .cast(pl.Float64),
  
   # Odstranění textu 'cm' ze vzdálenosti lampy
   pl.col("lamp_distance_cm")
     .str.replace_all(r"[^\d\-]", "")
     .cast(pl.Int64)
])


# Druhý průchod: Filtrace fyzikálních nesmyslů, odstranění duplicit a null hodnot
df_clean = df_clean.filter(
   # Vzdálenost lampy a úhel musí být v logických mezích
   (pl.col("lamp_distance_cm") >= 0) &
   (pl.col("angle_deg") >= 0) & (pl.col("angle_deg") <= 90)
).unique().drop_nulls(
   # Klíčová měření pro výpočet výkonu nesmí chybět
   subset=["voltage_v", "current_a"]
).with_columns(
   # Zaplnění chybějících poznámek
   pl.col("notes").fill_null("no notes")
)


print(f"Nový počet řádků po vyčištění a filtraci: {df_clean.height}")




# ==============================================================================
print("...3...")
# ==============================================================================
print("\n3. VÝPOČET REÁLNÉHO VÝKONU A SROVNÁNÍ S PŮVODNÍM SLOUPCEM")


df_clean = df_clean.with_columns(
   (pl.col("voltage_v") * pl.col("current_a")).alias("power_calc")
)


rozdily = df_clean.filter(
   (pl.col("power_w") - pl.col("power_calc")).abs() > 0.01
).select([
   "measurement_id", "voltage_v", "current_a", "power_w", "power_calc"
]).head(5)


print("Ukázka měření s chybami v původním sloupci power_w (rozdíl > 0.01 W):")
print(rozdily)




# ==============================================================================
print("...4...")
# ==============================================================================
print("\n4. PRŮMĚRNÝ VÝKON V ZÁVISLOSTI NA ÚHLU SKLONU PANELU")


analyza_uhlu = df_clean.group_by("angle_deg").agg(
   pl.col("power_calc").mean().alias("prumerny_vykon_W")
).sort("angle_deg")


print(analyza_uhlu)




# ==============================================================================
print("...5...")
# ==============================================================================
print("\n5. KORELACE MEZI INTENZITOU SVĚTLA (LUX) A VÝKONEM")


korelace = df_clean.select(
   pl.corr("light_intensity_lux", "power_calc").alias("korelacni_koeficient")
).item()


print(f"Pearsonův korelační koeficient: {korelace:.4f}")




# ==============================================================================
print("...6...")
# ==============================================================================
print("\n6. SROVNÁNÍ EFEKTIVITY PODLE TYPU PROSTŘEDÍ / POČASÍ")


analyza_prostredi = df_clean.group_by("weather").agg([
   pl.col("power_calc").mean().alias("prumerny_vykon_W"),
   pl.col("light_intensity_lux").mean().alias("prumerna_intenzita_lux")
]).sort("prumerny_vykon_W", descending=True)


print(analyza_prostredi)




# ==============================================================================
print("...7...")
# ==============================================================================
print("\n7. TOP 5 MĚŘENÍ S NEJVYŠŠÍM SKUTEČNÝM VÝKONEM (IDEÁLNÍ PODMÍNKY)")


top_podminky = df_clean.sort("power_calc", descending=True).select([
   "measurement_id", "weather", "light_intensity_lux",
   "angle_deg", "temperature_c", "power_calc"
]).head(5)


print(top_podminky)


# ==============================================================================
print("...8...")
# ==============================================================================


print("1. EXTRÉMNĚ VYSOKÉ HODNOTY LUX (INTENZITY):")
print(df_clean.filter(pl.col("light_intensity_lux") > 95000).select([
   "measurement_id", "weather", "light_intensity_lux", "power_w", "power_calc"
]))


# 2. Hledání extrémů u výkonu podle nápovědy (power_w > 10)
print("\n2. EXTRÉMNÍ HODNOTY VE SLOUPCI POWER_W (NAD 10 W):")
print(df_clean.filter(pl.col("power_w") > 10).select([
   "measurement_id", "weather", "light_intensity_lux", "voltage_v", "current_a", "power_w", "power_calc"
]))
