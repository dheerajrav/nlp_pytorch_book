# Generating a one-hot or binary representation of a sentence

from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

sentences = [
    "Ask not what your country can do for you - but what you can do for your country",
    "Freedom is not voluntarily given by the opressor; it must be demanded by the opressed."
]

one_hot_vectorizer = CountVectorizer(binary=True)
one_hot_vector = one_hot_vectorizer.fit(sentences)
transformed_sentences = one_hot_vector.transform(sentences)

vocab =  list(map(lambda x: x[0], sorted(one_hot_vector.vocabulary_.items())))

sns.heatmap(
    transformed_sentences.toarray(),
    annot=True, cbar=False,
    yticklabels = ["Sentence 1", "Sentence 2"],
    xticklabels = vocab
)
plt.show()
