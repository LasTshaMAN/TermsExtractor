from collections import Counter


class Corpus:
    def __init__(self, corpus):
        self.candidate_to_representative_mapping = Corpus.extract_representatives_for_candidates(corpus)
        self.candidate_to_df_mapping = Corpus.calculate_dfs_for(corpus)
        self.corpus_size = len(corpus)

    def get_df_for(self, candidate):
        return self.candidate_to_df_mapping.get(candidate, (1.0 / self.corpus_size))

    def get_representative_for(self, candidate):
        return self.candidate_to_representative_mapping.get(candidate)

    def get_terms(self):
        return self.candidate_to_df_mapping.keys()

    @staticmethod
    def extract_representatives_for_candidates(corpus):
        result = {}

        candidate_to_representative_mapping = {}
        for document in corpus:
            for candidate in document.get_candidates():
                representatives = document.get_representatives_for(candidate)
                if candidate not in candidate_to_representative_mapping:
                    candidate_to_representative_mapping[candidate] = []
                candidate_to_representative_mapping[candidate].extend(representatives)

        for candidate, representatives in candidate_to_representative_mapping.items():
            result[candidate] = Corpus.most_common(representatives)

        return result

    @staticmethod
    def most_common(lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]

    @staticmethod
    def calculate_dfs_for(corpus):
        result = {}

        for document in corpus:
            for candidate in document.get_candidates():
                if candidate in result:
                    result[candidate] += 1.0
                else:
                    result[candidate] = 1.0

        corpus_size = len(corpus)
        for candidate, count in result.items():
            result[candidate] = (count / corpus_size) * 100.0

        return result

