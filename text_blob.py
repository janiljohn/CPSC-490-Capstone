from textblob import TextBlob
from sklearn.metrics import classification_report
import nltk

nltk.download('treebank')

train_data = nltk.corpus.treebank.tagged_sents()[:3000]
test_data = nltk.corpus.treebank.tagged_sents()[3000:]

train_sentences = []
for sentence in train_data:
    for word, tag in sentence:
        train_sentences.append(word + '/' + tag)

train_sentence_blob = ' '.join(train_sentences)

tagger = TextBlob(train_sentence_blob)

correct = []
predicted = []

for sent in test_data:
    sent_words = [word for word, tag in sent]
    sent_text = ' '.join(sent_words)
    tagged_sent = tagger.tag(sent_text)
    for (_, tag), (_, predicted_tag) in zip(sent, tagged_sent):
        if tag is not None and predicted_tag is not None:
            correct.append(tag)
            predicted.append(predicted_tag)

report = classification_report(correct, predicted)
print(report)