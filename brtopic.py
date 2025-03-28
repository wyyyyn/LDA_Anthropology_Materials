import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer


file_path = r"processed_content_part1.xlsx"
df = pd.read_excel(file_path)

texts = df['content_cutted'].tolist()


embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

umap_model = UMAP(n_neighbors=15, n_components=5,min_dist=0.0,metric='cosine')

hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean', prediction_data=True)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer()

topic_model = BERTopic(
    embedding_model=embedding_model,  # Step 1 - Extract embeddings
    # umap_model=umap_model,  # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,  # Step 3 - Cluster reduced embeddings
    # vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
    # ctfidf_model=ctfidf_model,  # Step 5 - Extract topic words
    language="chinese",
    nr_topics=None,
)

#topic_model = BERTopic(embedding_model=embedding_model,language="chinese")
filtered_text = df["content_cutted"].tolist()
print(len(filtered_text))
topics, probs = topic_model.fit_transform(filtered_text)


df['BERTopic_Topic'] = topics
df['BERTopic_Probability'] = probs


all_topics = topic_model.get_topic_info()
all_topics.to_excel("bertopic_topics_info.xlsx", index=False)

output_file = r"bertopic_data_topic.xlsx"
df.to_excel(output_file, index=False)

# 可视化主题（交互式HTML文件）
embeddings = topic_model.c_tf_idf_.toarray()
topic_model.visualize_topics(custom_embeddings=embeddings).write_html("bertopic_topics.html")
#topic_model.visualize_topics().write_html("bertopic_topics.html")

# 查看主题结果
print(df.head())
print(all_topics.head())
