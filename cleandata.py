import pandas as pd
import re

# Load both datasets
final_df = pd.read_csv("final_fake_news.csv")
news_df = pd.read_csv("news.csv")

# STEP 1: Clean final_fake_news.csv
# Drop 'subject' column if it exists
if 'subject' in final_df.columns:
    final_df = final_df.drop(columns=['subject'])

# Remove special characters from title and text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^\w\s]', '', str(text))  # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

final_df['title'] = final_df['title'].apply(clean_text)
final_df['text'] = final_df['text'].apply(clean_text)

# STEP 2: Clean news.csv
news_df = news_df[['title', 'text', 'label']]  # Keep only these columns
news_df = news_df[news_df['label'].isin(['FAKE', 'REAL'])]  # Remove rows not in fake/real
news_df['label'] = news_df['label'].map({'FAKE': 0, 'REAL': 1})  # Convert label

# Clean text fields
news_df['title'] = news_df['title'].apply(clean_text)
news_df['text'] = news_df['text'].apply(clean_text)

# STEP 3: Merge the datasets
merged_df = pd.concat([final_df, news_df], ignore_index=True)

# Drop rows with missing label (just in case)
merged_df = merged_df.dropna(subset=['label'])

# Save the final merged and cleaned file
merged_df.to_csv("merged_news.csv", index=False)

print("Final clean dataset saved as 'merged_news_clean.csv'")
print("Final Columns:", merged_df.columns.tolist())
print("Total Rows:", len(merged_df))