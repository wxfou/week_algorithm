from sklearn.metrics import make_scorer, f1_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# load data 
def file_walker(file_path):
    file_list = []
    for root, dirs, files in os.walk(file_path): # a generator
        for fn in files:
            path = str(root+'/'+fn)
            file_list.append(path)
    return file_list
    
def read_txt(path, encoding):
    with open(path, 'r', encoding=encoding, errors='ignore') as f:
        lines = f.readlines()
    return lines
    
def extract_chinese(text):
    content = ' '.join(text)
    re_code = re.compile("\r|\n|\\s")
    re_punc = re.compile('[/s+/./!//_,$%^*()+/"/\']+|[+-——！★☆─◆‖□●█〓，。？、；~@#￥%……&*“”≡《》：（）]+') #remove punctation 
    chinese = re_punc.sub('', re_code.sub('', content))
    return chinese
    
def tokenize(text):
    word_list = []
    for word in jieba.cut(text, HMM=False):
        word_list.append(word)
    return word_list
    
def get_stopwords():
    with open('./data2018/Spam/stopwords.txt', 'r') as fhand:
        lines = fhand.readlines()
        stopwords = []
        for word in lines:
            stopwords.append(word.strip('\n'))
    return stopwords

def remove_stopwords(words):
    filtered = [word.lower() for word in words if(word.lower() not in stopwords)]
    return filtered
    
def get_fileid(filepath):    
    idx = filepath.split('/')[-1].split('..')[0]
    return idx
    
X = []
train_idx = []
for i in range(len(train_path_list)):
    content = extract_chinese(read_txt(train_path_list[i], encoding='gbk'))
    train_idx.append(get_fileid(train_path_list[i]))
    string = ' '.join(remove_stopwords(tokenize(content)))
    X.append(string)
    
def get_labels():
    with open(label_path, 'r') as fhand:
        label = {}
        lines = fhand.readlines()
        lines = lines[1:]
        for line in lines:
            line = line.strip('\n')
            label[line.split('\t')[1]] = line.split('\t')[0]
    # label['1'] = '1'
    return label

labels = get_labels()
y = []
for i in range(len(train_path_list)):
    y.append(labels[train_idx[i]])

count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore')
# count_vec.fit_transform(X) must split first
x_train, x_test, y_train, y_test\
    = train_test_split(X, y, test_size = 0.2)
x_train = count_vec.fit_transform(x_train)
x_test  = count_vec.transform(x_test)


nb = MultinomialNB()
lr = LogisticRegression()

knn = KNeighborsClassifier() 

models = [nb, lr, knn]


scorer = make_scorer(f1_score, 
                     greater_is_better=True, average="micro")
for m in models:
    score = cross_val_score(m, x_train, y_train, cv=5, scoring=scorer)
    print('{0}`s f1_score is:'.format(str(m).split('(')[0]))
    print(score)
