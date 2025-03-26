import os
import jieba
import re
import pandas as pd


txt_dir = r"./jilu"
stopwords_path = r"cn_stopwords.txt"
output_path = r"processed_content.xlsx"


with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = set(line.strip() for line in f)


def word_cut(content):
    content = re.sub(r'[^\u4e00-\u9fa5]', '', content)
    words = jieba.cut(content)
    filtered_words = [word for word in words if word not in stopwords and word.strip()]
    return " ".join(filtered_words)

data = []
for filename in os.listdir(txt_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(txt_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            processed_content = word_cut(content)
            data.append({'filename': filename, 'content_cutted': processed_content})


df = pd.DataFrame(data)


df.to_excel(output_path, index=False)

print(df.head())


