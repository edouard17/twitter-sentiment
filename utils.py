import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
from functions.contractions import CONTRACTION_MAP
import unicodedata

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

# finds user or an email address
user_links = "@([A-Za-z0-9_]+)?(\.[a-z]{2,4})?"
web_links_1 = "((https://|http://|www\.)?[a-zA-Z0-9._%+-]+\.[.a-z]{1,4}(([\/])([a-zA-Z0-9._%+-\/#]+)?)?) "
web_links_2 = "((https://|http://|www\.)?[a-zA-Z0-9._%+-]+\.[.a-z]{1,4}(([\/])([a-zA-Z0-9._%+-\/#]+)?)?) ?(?:(?=[\s.?!])[^.?!]*(?:[.?!].*)?)?$"

def remove_urls(text):
    text = re.sub(web_links_1, ' ', str(text)).strip()
    text = re.sub(web_links_2, ' ', str(text)).strip()
    text = re.sub(user_links, ' ', str(text)).strip()
    return text


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def normalize_tweet(text, html_stripping=True, contraction_expansion=True,
                    accented_char_removal=True, text_lower_case=True, 
                    text_lemmatization=True, special_char_removal=True, 
                    stopword_removal=True):
    
    # Remove user mentions and links
    text = remove_urls(text)
    # Strip HTML
    if html_stripping:
        text = strip_html_tags(text)
    # Remove accented characters
    if accented_char_removal:
        text = remove_accented_chars(text)
    # Expand contractions
    if contraction_expansion:
        text = expand_contractions(text)
    # Lowercase the text
    if text_lower_case:
        text = text.lower()
    # Remove extra newlines
    text = re.sub(r'[\r|\n|\r\n]+', '', text)
    # Insert spaces between special characters to isolate them
    special_char_pattern = re.compile(r'([{.(-)!}])')
    text = special_char_pattern.sub(" \\1 ", text)
    # Remove special characters
    if special_char_removal:
        text = remove_special_characters(text)
    # Remove extra whitespace
    text = re.sub(' +', ' ', text)
    # Remove stopwords
    if stopword_removal:
        text = remove_stopwords(text, is_lower_case=text_lower_case)
    # Save review
    return text



def text_to_indices(X, word_to_index, max_len):
    """
    Converts an array of reviews (strings) into an array of indices corresponding to words.
    """
    m = X.shape[0]
    
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):
        text_words = [x.lower() for x in X[i].split()]
        
        j = 0
        for w in text_words:
            if w in word_to_index:
                X_indices[i,j] = word_to_index[w]
            else:
                X_indices[i,j] = word_to_index['<UNK>']
            j += 1
    
    return X_indices



def read_glove_vecs(glove_file):
    print("Loading Glove Vectors")
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        count = 0
        average = np.zeros((50,))
        for line in f:
            line = line.strip().split()
            # make sure all vectors are 50-dimensional
            if len(np.array(line[1:], dtype=np.float64)) == 50:
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
                # sum the vectors so later we can get the average for unknown word
                average += np.array(line[1:], dtype=np.float64)
                # print evolution every 5,000 words
                count += 1
                if count % 5000 == 0:
                    print('.', sep=' ', end='', flush=True)
            else:
                continue
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
        # add <UNK> token for unknown word by computing the average
        words_to_index['<UNK>'] = len(words_to_index) + 1
        index_to_words[len(words_to_index) + 1] = '<UNK>'
        word_to_vec_map['<UNK>'] = average / len(index_to_words)
        print(" Done.",len(words_to_index)," words loaded!")
    return words_to_index, index_to_words, word_to_vec_map

