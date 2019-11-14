from flask import Flask, request, render_template
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("home.html")


@app.route("/execute", methods=["POST", "GET"])
def execute():
    if request.method == "POST":
        if request.form:
            text = request.form.get("text")
            token = sent_tokenize(text)
            # return render_template("result.html", token=token)

            # Word Tokenizing
            words = word_tokenize(text)

            # Removing StopWords

            stop_word = set(stopwords.words("english"))

            # Stopwords

            # Sentence Tokenizing
            word_tokens = word_tokenize(text)

            # Filtering sentence without stop words
            filtered_sentence = [w for w in word_tokens if not w in stop_word]

            # Filtered sentence

            # Stemming
            ps = PorterStemmer()

            stem = []
            for w in word_tokens:
                stem.append(ps.stem(w))

            # Part of speach Tagging
            tag = []
            word1 = nltk.word_tokenize(text)
            for i in word1:
                word1 = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(word1)
                tag.append(tagged)

            # Chunking
            custom_sent_tokenizer = PunktSentenceTokenizer(text)

            tokenized = custom_sent_tokenizer.tokenize(text)

            lemmatizer = WordNetLemmatizer()

            lemma = lemmatizer.lemmatize("default", pos="a")

            synonyms = []
            antonyms = []

            for syn in wordnet.synsets("angry"):
                for l in syn.lemmas():
                    synonyms.append(l.name())
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())

            return render_template(
                "result.html/",
                word=words,
                token=token,
                filterd=filtered_sentence,
                stem=stem,
                tag_list=tag,
                lemma=lemma,
                synonyms=synonyms,
                antonyms=antonyms,
            )

