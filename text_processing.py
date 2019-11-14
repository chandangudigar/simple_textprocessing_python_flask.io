import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

text = input("Enter the text >>>>")

# Sentence Tokenizing
print("Sentence Tokenizing:", sent_tokenize(text))

# Word Tokenizing
words = word_tokenize(text)
print("Word Tokenizing:", words)

# Removing StopWords

stop_word = set(stopwords.words("english"))

# Stopwords
print("StopWords:", stop_word)

# Sentence Tokenizing
word_tokens = word_tokenize(text)

# Filtering sentence without stop words
filtered_sentence = [w for w in word_tokens if not w in stop_word]

# Filtered sentence
print("Filtered sentence:", filtered_sentence)

# Stemming
ps = PorterStemmer()
print("After Stemming:")
for w in word_tokens:
    print(ps.stem(w))

# Part of speach Tagging

tag_list = [nltk.pos_tag(w) for w in words]
print("Part of speech Tagg:", tag_list)

for i in words:
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
    print(tagged)

# Chunking
custom_sent_tokenizer = PunktSentenceTokenizer(text)

tokenized = custom_sent_tokenizer.tokenize(text)


def process_content():
    try:

        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            chunked.draw()
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()
    except Exception as e:
        print(str(e))


process_content()

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("default", pos="a"))

synonyms = []
antonyms = []

for syn in wordnet.synsets("cool"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))
