import os
import re
import pandas as pd

# Directory containing English text files and output file path
txt_dir = r"./part1"
output_path = r"processed_content_part1.xlsx"

def preprocess_text(content):
    # Remove non-alphabetic characters and extra whitespace, then convert to lowercase
    content = re.sub(r'[^a-zA-Z\s]', '', content)
    content = re.sub(r'\s+', ' ', content)  # Normalize whitespace

    content = content.lower()
    # Remove any word that contains 'www'
    words = content.split()
    filtered_words = [word for word in words if ("www" and "http") not in word]
    return " ".join(filtered_words)

data = []
for filename in os.listdir(txt_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(txt_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            processed_content = preprocess_text(content)
            data.append({'filename': filename, 'content': processed_content})

df = pd.DataFrame(data)
df.to_excel(output_path, index=False)
print(df.head())
