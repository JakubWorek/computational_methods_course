from flask import Flask, render_template, request
from nltk.stem import PorterStemmer
from scipy import sparse
import scipy
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
import pickle
import string
import enchant

##################################################################################################################

# Load data

english_dict = enchant.Dict("en_US")
stopwords = stopwords.words("english")

with open('/media/jakub/Data/DEV/MOWNIT/lab_06/app/dataset/my_dataset.bin', 'rb') as file:
    dataset = pickle.load(file)
for i in range(len(dataset)):
    dataset[i]["id"] = i

with open('/media/jakub/Data/DEV/MOWNIT/lab_06/app/dataset/all_words.bin', 'rb') as file:
    all_words = pickle.load(file)
with open('/media/jakub/Data/DEV/MOWNIT/lab_06/app/dataset/words_per_article.bin', 'rb') as file:
    words_per_article = pickle.load(file)
with open('/media/jakub/Data/DEV/MOWNIT/lab_06/app/dataset/word_occurance.bin', 'rb') as file:
    word_occurance = pickle.load(file)
with open('/media/jakub/Data/DEV/MOWNIT/lab_06/app/dataset/my_dataset.bin', 'rb') as file:
    dataset = pickle.load(file)

all_words_indexes = dict()
for i, word in enumerate(all_words):
    all_words_indexes[word] = i

sparse_mat = sparse.load_npz("/media/jakub/Data/DEV/MOWNIT/lab_06/app/dataset/sparse_mat_default.npz")
sparse_mat_idf = sparse.load_npz("/media/jakub/Data/DEV/MOWNIT/lab_06/app/dataset/sparse_mat_idf.npz")
sparse_mat_normalized = sparse.load_npz("/media/jakub/Data/DEV/MOWNIT/lab_06/app/dataset/sparse_mat_normalized.npz")

N = len(dataset)
M = len(all_words)

##################################################################################################################

# Functions

def simplify_string(text: str) -> list:
    def my_filter(x):
        if len(x) < 2 or any(char.isdigit() for char in x) or not english_dict.check(x):
            return False
        return True

    ps = PorterStemmer()
    text = text.lower().translate(str.maketrans("\n", " ", string.punctuation)).split(" ")
    text = [word for word in text if word not in stopwords]
    text = list(filter(my_filter, text))
    text = list(map(ps.stem, text))
    return text

def create_single_vector(text: str):
    text = simplify_string(text)
    tmp = []
    for word in text:
        if word in all_words:
            tmp.append(word)
    text = tmp

    word_counter = dict()
    for word in text:
        if word not in word_counter:
            word_counter[word] = 1
        else:
            word_counter[word] += 1

    row = []
    col = []

    data = []
    for word in word_counter:
        row.append(0)
        col.append(all_words_indexes[word])
        data.append(word_counter[word])
    return sparse.csr_matrix((data, (row, col)), shape=(1, M))


def return_k_nearest_articles(searched_phrase, matrix_of_words, k = 10):
    q = create_single_vector(searched_phrase)
    q_norm = scipy.linalg.norm(q, axis=1)
    ans = []
    min_val = 0
    for i in range(N):
        col = matrix_of_words.getcol(i)
        val = q * col
        if not val:
            continue
        val = (val.data / (q_norm * scipy.linalg.norm(col, axis=0)))[0]
        if len(ans) < k:
            ans.append((i, val))
        elif val > min_val:
            ans.append((i, val))
            ans.sort(key = lambda x: x[1], reverse=True)
            ans.pop(-1)
            min_val = ans[-1][1]
    ans = [index[0] for index in ans]
    return ans

def return_k_nearest_articles_normalized(searched_phrase, k = 10):
    q = create_single_vector(searched_phrase)
    q_normalized = normalize(q, norm="l1", axis=1)
    ans_mat = q_normalized @ sparse_mat_normalized
    cx = ans_mat.tocoo()
    ans = []
    min_val = 0
    for i, val in zip(cx.col, cx.data):
        if len(ans) < k:
            ans.append((i, val))
        elif val > min_val:
            ans.append((i, val))
            ans.sort(key = lambda x: x[1], reverse=True)
            ans.pop(-1)
            min_val = ans[-1][1]
    ans = [index[0] for index in ans]
    return ans

def return_k_nearest_articles_svd(searched_phrase, svd_size, k = 10):
    print(f"Searching for: {searched_phrase}")
    q = create_single_vector(searched_phrase)
    print("Query created")
    q_normalized = normalize(q, norm="l1", axis=1)
    print("Query normalized")

    ans = []
    min_val = 0

    with open('/media/jakub/Data/DEV/MOWNIT/lab_06/dataset/sparse_svd_U_1000.bin', 'rb') as file:
        print("Loading U")
        svd_U = pickle.load(file)
        temp = q_normalized @ svd_U
        del svd_U  # Free up memory

    with open('/media/jakub/Data/DEV/MOWNIT/lab_06/dataset/sparse_svd_D_1000.bin', 'rb') as file:
        print("Loading D")
        svd_D = pickle.load(file)
        temp = temp @ sparse.diags(svd_D)
        del svd_D  # Free up memory

    with open('/media/jakub/Data/DEV/MOWNIT/lab_06/dataset/sparse_svd_V_1000.bin', 'rb') as file:
        print("Loading V")
        svd_V = pickle.load(file)
        ans_mat = temp @ svd_V
        del svd_V, temp  # Free up memory

    print("SVD loaded and matrix multiplied")

    ans_mat = sparse.csr_matrix(ans_mat)
    print("Matrix converted")
    cx = ans_mat.tocoo()

    for i, val in zip(cx.col, cx.data):
        if len(ans) < k:
            ans.append((i, val))
        elif val > min_val:
            ans.append((i, val))
            ans.sort(key = lambda x: x[1], reverse=True)
            ans.pop(-1)
            min_val = ans[-1][1]

    print(ans)
    ans = [index[0] for index in ans]
    return ans

##################################################################################################################

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/basic", methods=['GET', 'POST'])
def basicSearch():
    best_results=[]
    result = []
    if request.method == 'POST':
        best_results = return_k_nearest_articles_normalized(request.form['query'], 10)
        for i, index in enumerate(best_results):
            print(f"{i+1}: {dataset[index]['title']} | {dataset[index]['url']}")
        result = [(dataset[index]['title'],dataset[index]['url']) for index in best_results]
        
    return render_template('basicSearch.html', best_matches=result)

@app.route("/svd", methods=['GET', 'POST'])
def svdSearch():
    best_results=[]
    result = []
    if request.method == 'POST':
        best_results = return_k_nearest_articles_svd(request.form['query'], 1000)
        for i, index in enumerate(best_results):
            print(f"{i+1}: {dataset[index]['title']} | {dataset[index]['url']}")
        result = [(dataset[index]['title'],dataset[index]['url']) for index in best_results]
        
    return render_template('svdSearch.html', best_matches=result)

if __name__ == "__main__":
    app.run()
