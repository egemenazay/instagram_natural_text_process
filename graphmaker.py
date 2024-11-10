import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import numpy as np

df = pd.read_excel("k_means_cluster_results.xlsx")
cluster_texts_0 = df[df['Cluster'] == 0]['Document'].tolist()
cluster_texts_1 = df[df['Cluster'] == 1]['Document'].tolist()
cluster_texts_2 = df[df['Cluster'] == 2]['Document'].tolist()

text_0 = ' '.join(cluster_texts_0)
text_1 = ' '.join(cluster_texts_1)
text_2 = ' '.join(cluster_texts_2)

words_0 = re.findall(r'\w+', text_0)
words_1 = re.findall(r'\w+', text_1)
words_2 = re.findall(r'\w+', text_2)

unique_words_0 = set()
unique_words_1 = set()
unique_words_2 = set()

word_count_0 = len(words_0)
word_count_1 = len(words_1)
word_count_2 = len(words_2)

for word in words_0:
    unique_words_0.add(word)
for word in words_1:
    unique_words_1.add(word)
for word in words_2:
    unique_words_2.add(word)

unique_word_count_0 = len(unique_words_0)
unique_word_count_1 = len(unique_words_1)
unique_word_count_2 = len(unique_words_2)

wordcloud_0 = WordCloud(width=1000, height=500, background_color='white').generate(text_0)
wordcloud_1 = WordCloud(width=1000, height=500, background_color='white').generate(text_1)
wordcloud_2 = WordCloud(width=1000, height=500, background_color='white').generate(text_2)


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_0, interpolation='bilinear')
plt.title("Cluster 0")
plt.axis('off')

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_1, interpolation='bilinear')
plt.title("Cluster 1")
plt.axis('off')

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_2, interpolation='bilinear')
plt.title("Cluster 2")
plt.axis('off')

word_counts = [word_count_0,word_count_1,word_count_2]
unique_word_counts = [unique_word_count_0,unique_word_count_1,unique_word_count_2]

clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2']

bar_width = 0.35
indices = np.arange(len(clusters))

fig, ax = plt.subplots()

bar1 = ax.bar(indices, word_counts, bar_width, label='Total Word Count')
bar2 = ax.bar(indices + bar_width, unique_word_counts, bar_width, label='Unique Word Count')

ax.set_xlabel('Clusters')
ax.set_ylabel('Word Count')
ax.set_title('Comparison of Total and Unique Word Counts in Clusters')
ax.set_xticks(indices + bar_width / 2)
ax.set_xticklabels(clusters)
ax.legend()

plt.show()