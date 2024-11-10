import requests
import pandas as pd
import re
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
nlp = spacy.load("en_core_web_sm")  # İngilizce metin işlemek için sPacy modelini yüklüyorum


df = pd.read_excel("instagram_only_comments.xlsx")            # Bu satırlarda Excel dosyasındaki metinleri alıp
texts = df['text'].tolist()                                   # Python içerisinde Array'e dönüştürüyorum

texts[:] = [text if isinstance(text, str) else "" for text in texts]      # String gözükmeyen değerleri yok ediyorum

# Bu alanda metinler içerisindeki emojilerden kurtulan fonksiyonu oluşturuyorum
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# Veri setim İngilizce olduğu için İngilizce stopword'leri içeren verileri çekiyorum
stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
stop_words = set(stopwords_list.decode().splitlines())
stop_words.add('I')

# Tokenizasyon işlemi için 'Natural Language Processing Toolkit' üzerinden Tokenizasyon işlemi için gerekli metodları yüklüyorum
nltk.download('punkt_tab')

def preprocess_text(text):
    # Kelimeleri küçük harfe çeviriyorum
    text = text.lower()
    # Emojileri kaldırıyorum
    text = remove_emojis(text)
    # Noktalama işaretlerini kaldırıyorum
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenizasyon işlemini yapıyorum
    tokens = nltk.word_tokenize(text)
    # 'Stopword'leri çıkarıyorum
    tokens = [word for word in tokens if word not in stop_words]
    # Kelimeleri sadeleştiriyorum
    doc = nlp(" ".join(tokens))
    tokens = [token.lemma_ for token in doc]
    # Sayıları kaldırıyorum
    tokens = [re.sub(r'\d+', '', token) for token in tokens]
    # Boş Token'leri kaldırıyorum
    tokens = [token for token in tokens if token]
    return tokens

processed_texts = [preprocess_text(text) for text in texts]

# Her bir veri için token'leri tekrar birleştiriyorum
texts_as_strings = [" ".join(text) for text in processed_texts]

# Bütün işlemler sonrası boş kalan verileri kaldırıyorum
cleaned_list = pd.Series(texts_as_strings).dropna()
cleaned_list = cleaned_list[cleaned_list != ""].tolist()

# DataFrame oluşturup Excel dosyasına gönderiyorum
cleaned_texts_df = pd.DataFrame(cleaned_list)
cleaned_texts_df.to_excel("instagram_cleanedup_data.xlsx", index=False)

# Hazırladığım verileri kullanarak TF-IDF matrisini oluşturuyorum
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_list)

# TF-IDF matrisini kullanarak K-Kümeleme yöntemiyle 3 kümeye ayırıyorum
num_clusters = 3  # You can adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# Küme verilerini alıyorum
labels = kmeans.labels_

# Kümelenmiş verileri DataFrame oluşturup Excel'e aktarıoyrum
cluster_df = pd.DataFrame({
    "Document": cleaned_list,
    "Cluster": labels
})
cluster_df.to_excel("k_means_cluster_results.xlsx", index=False)





