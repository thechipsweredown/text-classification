import fasttext
# classifier = fasttext.supervised('data.txt', 'model',label_prefix='__label__')
classifier = fasttext.load_model('model.bin', label_prefix='__label__')
text = ['chào bạn']
labels = classifier.predict(text)
print(labels)