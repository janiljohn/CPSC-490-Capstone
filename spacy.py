import nltk
import spacy
from sklearn.metrics import classification_report

nlp = spacy.load('en_core_web_sm')
train_data = nlp.pipe([' '.join(sent) for sent in nltk.corpus.treebank.sents()[:3000]])
test_data = nlp.pipe([' '.join(sent) for sent in nltk.corpus.treebank.sents()[3000:]])

train_docs = list(train_data)
train_labels = []
for doc in train_docs:
    label = []
    for token in doc:
        label.append((token.text, token.pos_))
    train_labels.append(label)
tagger = nltk.tag.UnigramTagger(train_labels)

correct = []
predicted = []

for sent in test_data:
    for token in sent:
        if token.pos_ is not None:
            predicted_tag = tagger.tag([token.text])[0][1]
            if predicted_tag is not None:
                correct.append(token.pos_)
                predicted.append(predicted_tag)

report = classification_report(correct, predicted)
print(report)