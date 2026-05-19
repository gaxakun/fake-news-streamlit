import re

# Custom stopword list (common English words)
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'a', 'an', 'and', 'or', 'but', 'if',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'without', 'after',
    'upon', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'that',
    'the', 'this', 'those', 'been', 'has', 'have', 'having', 'is', 'am', 'are', 'was', 'were',
    'be', 'being', 'do', 'did', 'doing', 'does', 'would', 'could', 'should', 'might', 'must',
    'where', 'when', 'why', 'how', 'very', 'just', 'don', 'don\'t', 'doesn\'t', 'didn\'t'
])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def stem_word(word):
    # Simple Porter-like stemmer for common suffixes
    if word.endswith('ing'):
        word = word[:-3]
    elif word.endswith('ed'):
        word = word[:-2]
    elif word.endswith('s'):
        word = word[:-1]
    return word

def preprocess_text(text):
    text = clean_text(text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    words = [stem_word(w) for w in words]
    return ' '.join(words)
