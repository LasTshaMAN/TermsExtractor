import nltk
import re


class Document:
    def __init__(self, text):
        self.candidate_to_representatives_mapping = Document.extract_candidates_from(text)

    def get_tf_for(self, candidate):
        return len(self.get_representatives_for(candidate))

    def get_representatives_for(self, candidate):
        return self.candidate_to_representatives_mapping.get(candidate, [])

    def get_candidates(self):
        return self.candidate_to_representatives_mapping.keys()

    @staticmethod
    def extract_candidates_from(text):
        result = {}

        tokens = nltk.word_tokenize(text)
        stemmer = nltk.PorterStemmer()
        for token in tokens:
            normalized_token = re.sub('[^a-zA-z]', '', stemmer.stem(token.lower()))
            if len(normalized_token) > 2:
                if normalized_token not in result:
                    result[normalized_token] = []
                result[normalized_token].append(token)

        return result
