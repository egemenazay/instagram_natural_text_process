import pandas as pd
import re
import nltk
import spacy

nlp = spacy.load("en_core_web_sm")

df = pd.read_excel("instagram_only_comments.xlsx")            # Bu satırlarda Excel dosyasındaki metinleri alıp
texts = df['text'].tolist()                                   # Python içerisinde Array'e dönüştürüyorum

texts[:] = [text if isinstance(text, str) else "" for text in texts]       # String gözükmeyen değerleri yok ediyorum

combined_text = " ".join(texts)

# Bu alanda metinler içerisindeki emojilerden kurtulan metodu oluşturuyorum
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

combined_text = str(combined_text).lower()                          # Metinler içindeki bütün kelimeleri küçük harfe çeviriyorum
combined_text = remove_emojis(combined_text)                        # Metin içindeki emojileri yok ediyorum
combined_text = re.sub(r'[^\w\s]', '', combined_text)  # Noktalama işaretlerini kaldırıyorum

# Tokenization işlemi
nltk.download('punkt_tab')
tokens = nltk.word_tokenize(combined_text)

# Kelimeleri "Lemmatization" yani sadeleştirme işelmi. ÖRNEK: "Wheres" -> "where" "s"
doc = nlp(" ".join(tokens))
tokens = [token.lemma_ for token in doc]

# Stopword çıkarma işlemi
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.add('I')
tokens = [word for word in tokens if word not in stop_words]


tokens = [re.sub(r'\d+', '', token) for token in tokens]

tokens = [token for token in tokens if token != ""]

tokensdf = pd.DataFrame(tokens, columns=['Tokens'])
tokensdf.to_excel("instagram_post_comments_preprocessed.xlsx", index=False)

print(tokens)

