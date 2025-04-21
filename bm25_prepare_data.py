import os
import re
import math
import pickle
import string
import nltk
from collections import Counter, defaultdict
from datasets import load_dataset
from tqdm import tqdm
from nltk.corpus import stopwords, words

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("words")

#intialise regular expression to find URLs and save it to url_pattern
url_pattern = re.compile(r"(http://\S+|https://\S+|www\.\S+)")
#intialise regular expression to find contractions and save it to contraction_pattern
contraction_pattern = re.compile(r"'s|'m|'t|'re|'ve|'ll|'d")

#get a list of stopwords in the english dictionary using nltk and save it as set to stopwords
stopwords = set(stopwords.words("english"))
#get a list of the punctuations using string library and save it as set to punctuations eg !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
punctuations = set(string.punctuation)
#get the list of english words from nltk and convert them to lowercase and save the it as a set to english_vocab
english_vocab = set(word.lower() for word in words.words())
#creating a stemmer object using nltk using PorterStemmer
stemmer = nltk.stem.PorterStemmer()

#preprocessing the text  
def tokenize(text):
    #remove leading and trailing white spaces in text
    text = text.strip()

    #remove double spaces within text
    #splits text into words and then removes any extra white spaces and then join the words with no extra white spaces together
    text = " ".join(text.split())

    #remove URLs from text 
    text = url_pattern.sub("", text)
    
    #remove contractions from text
    text = contraction_pattern.sub("", text)

    #remove apostrophe and quotations from text 
    text = text.replace("’", "").replace("'", "")

    #tokenizes the text using nltk library
    tokens = nltk.word_tokenize(text)

    #processed tokens will be saved in this list
    processed_tokens = []
    #loop over each token within tokens
    for token in tokens:
        #turn current token to lowercase and then apply stemmer on it and save it to token
        token = stemmer.stem(token.lower())
        #check if all characters in the text are letters then check if token is not in list of stop words then check if token is not a puntucation and is a english word
        if token.isalpha() and token not in stopwords and token not in punctuations and token in english_vocab:
            #append token to processed_tokens
            processed_tokens.append(token)
    #return the list of processed tokens
    return processed_tokens


def process_example(example):

    tokens = tokenize(example["text"])
    example["tokens"] = tokens
    example["length"] = len(tokens)

    return example


def compute_precomputed_data(dataset, num_proc=4):
    print("Processing documents in parallel...")

    processed_dataset = dataset.map(process_example, num_proc=num_proc)
    

    docs = [doc for doc in tqdm(processed_dataset, total=len(processed_dataset), desc="Collecting processed docs")]
    

    for doc in tqdm(docs, desc="Computing document frequencies per doc"):
        doc["freqs"] = dict(Counter(doc["tokens"]))
    

    total_length = sum(doc["length"] for doc in docs)
    avgdl = total_length / len(docs)
    print(f'Average document length: {avgdl}')
    

    df = defaultdict(int)
    for doc in tqdm(docs, desc="Computing overall document frequencies"):
        for term in set(doc["tokens"]):
            df[term] += 1
    print(f'Total vocab size: {len(df)}')
    print(f'Number of documents: {len(docs)}')
    

    N = len(docs)
    idf_values = {}
    for term, freq in df.items():
        idf_values[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
    
    return docs, avgdl, idf_values

def save_precomputed_data(docs, avgdl, idf_values, directory="precomputed_data"):
    print("Saving data!")
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)
    with open(os.path.join(directory, "avgdl.pkl"), "wb") as f:
        pickle.dump(avgdl, f)
    with open(os.path.join(directory, "idf_values.pkl"), "wb") as f:
        pickle.dump(idf_values, f)
    print("Data saved!")

def load_precomputed_data(directory="precomputed_data"):
    print("Loading data!")
    with open(os.path.join(directory, "docs.pkl"), "rb") as f:
        docs = pickle.load(f)
    with open(os.path.join(directory, "avgdl.pkl"), "rb") as f:
        avgdl = pickle.load(f)
    with open(os.path.join(directory, "idf_values.pkl"), "rb") as f:
        idf_values = pickle.load(f)
    return docs, avgdl, idf_values

if __name__ == "__main__":
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

    shuffled_dataset = ds.shuffle(seed=42)
    subset_size = min(640000, len(shuffled_dataset))
    dataset = shuffled_dataset.select(range(subset_size))
    
    if not os.path.exists("precomputed_data/docs.pkl"):
        docs, avgdl, idf_values = compute_precomputed_data(dataset, num_proc=16)
        save_precomputed_data(docs, avgdl, idf_values)
    else:
        docs, avgdl, idf_values = load_precomputed_data()

    k1 = 1.5
    b = 0.75

    def bm25_score(query, doc):
        query_tokens = tokenize(query)
        score = 0.0
        for term in query_tokens:
            if term not in doc["freqs"]:
                continue
            f = doc["freqs"][term]
            term_idf = idf_values.get(term, 0)
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * (doc["length"] / avgdl))
            score += term_idf * (numerator / denominator)
        return score

    query = "car"

    scores = []
    for doc in docs:
        score = bm25_score(query, doc)
        scores.append((doc['id'], doc['title'], doc['url'], score))

    scores.sort(key=lambda x: x[3], reverse=True)

    for doc_id, title, url, score in scores[:100]:
        print(f"Document {doc_id} - {title}: BM25 Score = {score:.4f}, URL: {url}")
