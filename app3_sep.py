#importing necessary libraries
from flask import Flask, render_template, request, jsonify
import os, re, math, pickle, string, threading, time
import nltk
from nltk.corpus import stopwords, words
import nltk.stem
from augment import generate_augmented_queries# generate_answer
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt_tab')

#intialising Flask and saving it to app
app = Flask(__name__)

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

#fucntion to load precomputed data from specified directory loads documents, average document length and idf_values and returns these
def load_precomputed_data(directory="precomputed_data"):
    print("Loading data!")
    with open(os.path.join(directory, "docs.pkl"), "rb") as f:
        docs = pickle.load(f)
    with open(os.path.join(directory, "avgdl.pkl"), "rb") as f:
        avgdl = pickle.load(f)
    with open(os.path.join(directory, "idf_values.pkl"), "rb") as f:
        idf_values = pickle.load(f)
    return docs, avgdl, idf_values

#call the load_precomputed_data and save the outputs to the respective variables
docs, avgdl, idf_values = load_precomputed_data()

#fucntion to build dictionary of inverted indexes based on the documents passed
def build_inverted_index(docs):
    #dictionary that will store the inverted indexes based on the documents passed values are tokens and keys are document id
    inverted_index = {}
    #for looop to loop over idx and doc from docs 
    for idx, doc in enumerate(docs):
        #for loop to loop over term frequencies
        for token in doc["freqs"]:
            #check if token is not in inverted_index dictionary 
            if token not in inverted_index:
                #intialise empty list for value token within the dictionary 
                inverted_index[token] = set()
            #add keys to the respective values set within the dictionary 
            inverted_index[token].add(idx)
    #return dictionary containing values which are tokens and keys are set of document id for documents passed 
    return inverted_index

#call the function build_inverted_index and pass docs and save output to inverted_index
inverted_index = build_inverted_index(docs)

#intialise k1 as 1.5
k1 = 1.5
#intialise b as 0.75
b = 0.75

#function to calcualte bm25 score for the query tokens and doc
def bm25_score_tokens(query_tokens, doc):
    #intialise variable score and set it to 0.0
    score = 0.0
    #loop over each term within the query_tokens
    for term in query_tokens:
        #check if term is not in term frequencies if not then move to the next term 
        if term not in doc["freqs"]:
            continue
        #get the term frequncy of the current term
        f = doc["freqs"][term]
        #get the idf of the current term if it exists if not set it to 0 
        term_idf = idf_values.get(term, 0)
        #compute numerator for bm25 score using f and k1
        numerator = f * (k1 + 1)
        #compute denominator for bm25 score using f, k1, b document length and average document length 
        denominator = f + k1 * (1 - b + b * (doc["length"] / avgdl))
        #compute the score using term_idf, numerator and denominator
        score += term_idf * (numerator / denominator)
    #return the bm25 score for the query token within current doc 
    return score

search_jobs = {}

#fucntion to rank documents for current query
def search_worker(query):
    #call function generate_augmented_queries pass query save output to augmented_queries
    augmented_queries = generate_augmented_queries(query)

    search_jobs[query]['augmented'] = augmented_queries
    #add current query and augmented queries to all_queries as a list 
    all_queries = [query] + augmented_queries
    
    #tokenize all queries and save it to tokenized_queries
    tokenized_queries = [tokenize(q) for q in all_queries]
    
    #intialise empty set to store union_tokens
    union_tokens = set()
    #loop over each token within tokenized_queries
    for tokens in tokenized_queries:
        #update the union_tokens with tokens (prevents duplicate tokens within union_tokens)
        union_tokens.update(tokens)
    
    #intialise empty set to store candidate_docs which is set of documents which are candidates for the given term 
    candidate_docs = set()
    #loop over each token within union_tokens
    for token in union_tokens:
        #check if token exists within inverted_index dictionary 
        if token in inverted_index:
            #update the candidate_docs with the current token document id's
            candidate_docs.update(inverted_index[token])
    #inorder to make the set a indexable convert it to a list
    candidate_docs = list(candidate_docs)
    
    #check if no candidate documents are found for the users query 
    if not candidate_docs:
        #set progrees bar for the query to 100 
        search_jobs[query]['progress'] = 100
        #store an empty result list so we can display to the user nothing has been found 
        search_jobs[query]['results'] = []
        return
    
    #create empty array which will store scores 
    scores = []
    #get the legnth of candidate_docs and save it to total_candidate_length
    total_candidate_length = len(candidate_docs)
    
    #for loop to loop over i,doc_idx within candidate_docs
    for i, doc_idx in enumerate(candidate_docs):
        #get the current doc at doc_idx and store it in doc
        doc = docs[doc_idx]
        #get the maximum bm25 score for current tokens in tokenized_queries, and current doc at doc_idx by calling bm25_score_tokens 
        #and apply max on the output then apply max and save the max bm25 score to aggregated_score
        aggregated_score = max(bm25_score_tokens(tokens, doc) for tokens in tokenized_queries)
        #check if aggregated_score is greater than 0
        if aggregated_score > 0:
            #to scores apped a tuple of document id, document title, document url and its bm25 score 
            scores.append((doc['id'], doc['title'], doc['url'], aggregated_score))

        #code to compute progress of current search job 
        if (i + 1) % 100 == 0 or (i + 1) == total_candidate_length:
            #calculate progress in percentage
            progress = int((i + 1) / total_candidate_length * 100)
            #update the progress for the users current query
            search_jobs[query]['progress'] = progress
    #sort scores by the third element and have it in descending order where the first index has the highest score 
    scores.sort(key=lambda x: x[3], reverse=True)
    
    #set the results for the query to scores list which is in descending order where the first index has the highest score 
    search_jobs[query]['results'] = scores


#main page 
@app.route("/")
def index():
    #render index.html which is the main page
    return render_template("index.html")

#Post method that will take user query and return output from the retrieval model 
@app.route("/start_search", methods=["POST"])
def start_search():
    #get the query the user passed
    query = request.form.get("query")
    #return error if user passed empty query
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    #intialise a new search job with results set to none and progress set to 0 
    search_jobs[query] = {"progress": 0, "results": None}

    #start new thread that will run the search
    thread = threading.Thread(target=search_worker, args=(query,))
    thread.start()

    #return json response which shows the status of the search with the query
    return jsonify({"status": "started", "query": query})

#route to get the progress of the current search
@app.route("/progress")
def progress():
    #get the query 
    query = request.args.get("query")
    #if query is not found within search jobs then return 0 progress
    if query not in search_jobs:
        return "data:0\n\n", 200, {'Content-Type': 'text/event-stream'}
    #get the progress of the query
    progress = search_jobs[query]['progress']
    #return the progress of the query
    return f"data:{progress}\n\n", 200, {'Content-Type': 'text/event-stream'}

#route to show the search results 
@app.route("/results")
def results():
    #get the query and page number 
    query = request.args.get("query")
    # gen_ans = generate_answer(query)
    page = request.args.get("page", 1, type=int)
   #if query is not found within search jobs or query results are None then return results not ready message
    if query not in search_jobs or search_jobs[query]['results'] is None:
        return "Results not ready", 202
    
    #get the results for the specific query and set up the pages to display 10 results per page and have 10 pages in total
    scores = search_jobs[query]['results']
    augmented_queries = search_jobs[query].get('augmented', [])
    total_results = len(scores)
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    paged_scores = scores[start:end]
    total_pages = math.ceil(total_results / per_page)
    total_pages = min(total_pages, 10)
    #if there are no results found then show empty results page else display the results 10 results per page with 10 total pages
    if total_results == 0:
        return render_template("results.html", results=[], query=query, page=page, total_pages=total_pages, augmented_queries=augmented_queries)
    else:
        return render_template("results.html", results=paged_scores, query=query, page=page, total_pages=total_pages, augmented_queries=augmented_queries)

if __name__ == "__main__":
    app.run(debug=True, port=5000)