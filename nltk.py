import nltk
from sklearn.metrics import classification_report

nltk.download('treebank')

train_data = nltk.corpus.treebank.tagged_sents()[:3000]
test_data = nltk.corpus.treebank.tagged_sents()[3000:]

tagger = nltk.tag.UnigramTagger(train_data)

correct = []
predicted = []

for sent in test_data:
    for word, tag in sent:
        predicted_tag = tagger.tag([word])[0][1]
        if tag is not None and predicted_tag is not None:
            correct.append(tag)
            predicted.append(predicted_tag)

report = classification_report(correct, predicted)
print(report)