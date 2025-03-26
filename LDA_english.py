


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.lda_model

# Load the processed Excel file
file_path = r"processed_content_part1.xlsx"
df = pd.read_excel(file_path)

# Vectorize the text using CountVectorizer with English stop words
n_features = 500
tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df=0.5,
                                min_df=0.2)
tf = tf_vectorizer.fit_transform(df.content)

# Set the number of topics
n_topics = 4
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
                                random_state=0)
lda.fit(tf)

def print_top_words(model, feature_names, n_top_words):
    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        words = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        topic_words.append(words)
        print(words)
    return topic_words

n_top_words = 25
tf_feature_names = tf_vectorizer.get_feature_names_out()
topic_words = print_top_words(lda, tf_feature_names, n_top_words)

# Assign each document to its dominant topic
topics = lda.transform(tf)
topic_assignments = [list(t).index(np.max(t)) for t in topics]
df['topic'] = topic_assignments
print(df.head())

# Save the DataFrame with topic assignments to an Excel file
df.to_excel(r"data_topic_part1.xlsx", index=False)

# Optional: Visualize topics using pyLDAvis

panel = pyLDAvis.lda_model.prepare(lda, tf, tf_vectorizer)
pyLDAvis.save_html(panel, 'lda_visualization_part1.html')

# Evaluate and plot perplexity and score for various topic numbers
plexs = []
scores = []
n_max_topics = 16
for i in range(1, n_max_topics):
    lda_temp = LatentDirichletAllocation(n_components=i, max_iter=50,
                                           learning_method='batch',
                                           learning_offset=50,
                                           random_state=0)
    lda_temp.fit(tf)
    plexs.append(lda_temp.perplexity(tf))
    scores.append(lda_temp.score(tf))

# used to choose the number of topics
n_t = 15
x = list(range(1, n_t))
plt.plot(x, plexs[1:n_t])
plt.xlabel("Number of Topics")
plt.ylabel("Perplexity")
plt.title("Perplexity vs. Number of Topics")
plt.show()

# Plot score vs. number of topics
plt.plot(x, scores[1:n_t])
plt.xlabel("Number of Topics")
plt.ylabel("Score")
plt.title("Score vs. Number of Topics")
plt.show()