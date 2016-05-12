import math
from Corpus import Corpus


class ClusteredCorpus:
    def __init__(self, clustered_corpus):
        self.corpora = []
        for cluster in clustered_corpus:
            corpus = Corpus(cluster)
            self.corpora.append(corpus)
        if len(self.corpora) < 2:
            raise ValueError("clustered_corpus argument is not clustered")

        self.candidate_to_cu_mapping = self.calculate_cus_for()

    def get_cu_for(self, candidate):
        return self.candidate_to_cu_mapping.get(candidate, 0.0)

    def get_dfs_in_each_cluster_for(self, candidate):
        result = []

        for corpus in self.corpora:
            result.append(corpus.get_df_for(candidate))

        return result

    def calculate_cus_for(self):
        result = {}

        candidates = set()
        for corpus in self.corpora:
            candidates.update(corpus.get_terms())

        candidate_to_dfs_mapping = {}
        for candidate in candidates:
            candidate_to_dfs_mapping[candidate] = []
            for corpus in self.corpora:
                df = corpus.get_df_for(candidate)
                candidate_to_dfs_mapping[candidate].append(df)

        amount_of_clusters = len(self.corpora)
        for candidate, dfs in candidate_to_dfs_mapping.items():
            max_df = max(dfs)
            cu = 0.0
            for df in dfs:
                if not math.isclose(max_df, df, rel_tol=0.01):
                    cu += ((max_df - df) + (8.0 * math.log(2.0 + max_df / df, 2.0))) / 9.0
            cu /= (amount_of_clusters - 1)
            result[candidate] = cu

        return result
