import pytest
from information_retrieval import *


def test_preprocess_text():
    result = preprocess_text('This is the test text')
    assert isinstance(result, str)


def test_create_tfidf_features():
    data, vectorizer = create_tfidf_features(['For the Smith-Hughes , George-Barden , and National Defense Act of 1958 , the cumulative total of Federal expenditures in 42 years was only about $740 million.',
    'No comparable measures are available of enrollments and expenditures for private vocational education training.', 
    'There are a great number and variety of private commercial schools , trade schools and technical schools',
    'These schools are intended to provide the facilities and specialized curriculum that would not be possible for very small schoo'])
    assert data.shape[0] > 0 and data.shape[1] > 0


def test_get_similar_documents():
    output = get_similar_documents('education training')
    assert len(output) > 0
