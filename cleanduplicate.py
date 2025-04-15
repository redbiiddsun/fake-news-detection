import pandas as pd

# Load the dataset
df = pd.read_csv("merged_news.csv")

# Remove duplicate rows
df_clean = df.drop_duplicates()

# Save the cleaned dataset
df_clean.to_csv("merged_news_clean_v2.csv", index=False)

print(f"Duplicates removed. Clean file saved as 'merged_news_clean_v2.csv'")
print(f"Original rows: {len(df)}")
print(f"After cleaning: {len(df_clean)}")