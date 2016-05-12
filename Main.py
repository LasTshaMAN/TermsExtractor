import math
from Corpus import Corpus
from ClusteredCorpus import ClusteredCorpus
from Document import Document

from prettytable import PrettyTable
import os
import operator


def main():
    clustered_corpus_path = 'clustered_corpus'
    clustered_corpus = read_clustered_corpus(clustered_corpus_path)
    corpus = merge_clustered_corpus_into_a_single_corpus(clustered_corpus)

    target_file_path = 'target.txt'
    text = read_text_file(target_file_path)
    document = Document(text)

    corpus = Corpus(corpus)
    clustered_corpus = ClusteredCorpus(clustered_corpus)

    candidate_to_rank_mapping = {}
    candidate_to_params_mapping = {}
    candidate_to_dfs_in_each_cluster_mapping = {}

    for candidate in document.get_candidates():
        tf = math.log(1.0 + document.get_tf_for(candidate), 10.0)
        # tf = document.get_tf_for(candidate)
        idf = math.log(1.0 + 1.0 / corpus.get_df_for(candidate), 2.0)
        cu = clustered_corpus.get_cu_for(candidate)

        rank = cu
        # rank = tf * cu
        # rank = tf * idf

        dfs_in_each_cluster = clustered_corpus.get_dfs_in_each_cluster_for(candidate)

        candidate_representative = corpus.get_representative_for(candidate)
        candidate_to_rank_mapping[candidate_representative] = rank
        candidate_to_params_mapping[candidate_representative] = (tf, idf, cu)
        candidate_to_dfs_in_each_cluster_mapping[candidate_representative] = dfs_in_each_cluster

    table = generate_table_based_on(
        candidate_to_rank_mapping,
        candidate_to_params_mapping,
        candidate_to_dfs_in_each_cluster_mapping
    )
    table.align = 'l'
    print(table)


def read_clustered_corpus(path):
    result = []

    for directory in os.listdir(path):
        cluster = []
        for file in os.listdir(os.path.join(path, directory)):
            text_file = read_text_file(os.path.join(path, directory, file))
            document = Document(text_file)
            cluster.append(document)
        result.append(cluster)

    return result


def merge_clustered_corpus_into_a_single_corpus(clustered_corpus):
    result = []

    for cluster in clustered_corpus:
        result.extend(cluster)

    return result


def read_text_file(path):
    return open(path, 'r', encoding='utf-8').read()


def generate_table_based_on(
        candidate_to_rank_mapping,
        candidate_to_params_mapping,
        candidate_to_dfs_in_each_cluster_mapping
):
    result = PrettyTable(['Candidate', 'Rank', 'TF', 'IDF', 'CU', 'DF_1', 'DF_2', 'DF_3'])

    sorted_mapping = sorted(candidate_to_rank_mapping.items(), key=operator.itemgetter(1), reverse=True)
    for candidate, rank in sorted_mapping:
        params = candidate_to_params_mapping[candidate]
        tf = params[0]
        idf = params[1]
        cu = params[2]
        dfs_in_each_cluster = candidate_to_dfs_in_each_cluster_mapping[candidate]
        df_1 = dfs_in_each_cluster[0]
        df_2 = dfs_in_each_cluster[1]
        df_3 = dfs_in_each_cluster[2]
        result.add_row([candidate, str(rank), str(tf), str(idf), str(cu), str(df_1), str(df_2), str(df_3)])

    return result


if __name__ == "__main__":
    main()
