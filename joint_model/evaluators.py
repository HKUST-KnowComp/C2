import json
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np

def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

# https://stackoverflow.com/questions/10301000/python-connected-components
def getRoots(aNeigh):
    def findRoot(aNode,aRoot):
        while aNode != aRoot[aNode][0]:
            aNode = aRoot[aNode][0]
        return (aNode,aRoot[aNode][1])
    myRoot = {}
    for myNode in aNeigh.keys():
        myRoot[myNode] = (myNode,0)
    for myI in aNeigh:
        for myJ in aNeigh[myI]:
            (myRoot_myI,myDepthMyI) = findRoot(myI,myRoot)
            (myRoot_myJ,myDepthMyJ) = findRoot(myJ,myRoot)
            if myRoot_myI != myRoot_myJ:
                myMin = myRoot_myI
                myMax = myRoot_myJ
                if  myDepthMyI > myDepthMyJ:
                    myMin = myRoot_myJ
                    myMax = myRoot_myI
                myRoot[myMax] = (myMax,max(myRoot[myMin][1]+1,myRoot[myMax][1]))
                myRoot[myMin] = (myRoot[myMax][0],-1)
    myToRet = {}
    for myI in aNeigh:
        if myRoot[myI][0] == myI:
            myToRet[myI] = []
    for myI in aNeigh:
        myToRet[findRoot(myI,myRoot)[0]].append(myI)
    return myToRet


def score2clusters(triplets, scores):
    assert len(triplets) == len(scores)
    new_triplets = [[triplets[step][0], triplets[step][1], score] for step, score in enumerate(scores)]
    return links2clusters(new_triplets)

def score2clusters_new(triplets, scores):
    assert len(triplets) == len(scores)
    new_triplets = [[triplets[step][0], triplets[step][1], score] for step, score in enumerate(scores)]
    return links2clusters_new(new_triplets)

def links2clusters(triplets):
    mentions = {}
    # add mention pairs to mentions
    # Also should include singletons so that the scores are more easy to calculate
    all_mentions = set()
    linked_mentions = set()
    for triplet in triplets:
        all_mentions.add(triplet[0])
        all_mentions.add(triplet[1])
        if triplet[2] == 1:
            linked_mentions.add(triplet[0])
            linked_mentions.add(triplet[1])

            if triplet[0] in mentions:
                mentions[triplet[0]].append(triplet[1])
            else:
                mentions[triplet[0]] = [triplet[1]]

            if triplet[1] in mentions:
                mentions[triplet[1]].append(triplet[0])
            else:
                mentions[triplet[1]] = [triplet[0]]

    singletons = [{mention} for mention in (all_mentions - linked_mentions)]

    result_sets = [set(l) for l in getRoots(mentions).values()]
    result_sets.extend(singletons)
    return result_sets

def partition(S, responds):
    # There is an assumption that the responds is a graph generated from links and with singletons
    return [S & r for r in responds if len(S & r) > 0]

def MUC_score(keys, responds):

    r_nominator = 0
    r_denominator = 0

    p_nominator = 0
    p_denominator = 0

    for key in keys:
        size_of_S = len(key)
        size_of_pS = len(partition(key, responds))

        r_nominator += (size_of_S - size_of_pS)
        r_denominator += (size_of_S - 1)

    for respond in responds:
        size_of_S_prime = len(respond)
        size_of_pS_prime = len(partition(respond, keys))

        p_nominator += (size_of_S_prime - size_of_pS_prime)
        p_denominator += (size_of_S_prime - 1)

    if p_denominator == 0:
        p = 0
    else:
        p = p_nominator / p_denominator

    if r_denominator == 0:
        r = 0
    else:
        r = r_nominator / r_denominator
    f = f1_score(p, r)
    return p, r, f


def phi4(c1, c2):
    return 2 * len(c1 & c2) / (len(c1) + len(c2))

def Bcube_score(keys, responds):
    # List of Sets

    r_nominator = 0
    r_denominator = 0

    p_nominator = 0
    p_denominator = 0

    for key in keys:
        denominator = len(key)
        pS = partition(key, responds)
        for part in pS:
            r_nominator += len(part) / denominator * len(part)
            r_denominator += len(part)

    if r_denominator == 0:
        r = 0
    else:
        r = r_nominator / r_denominator

    for respond in responds:
        denominator = len(respond)
        pS = partition(respond, keys)
        for part in pS:
            p_nominator += len(part) / denominator * len(part)
            p_denominator += len(part)

    if p_denominator == 0:
        p = 0
    else:
        p = p_nominator / p_denominator

    return p, r, f1_score(p, r)

def Ceaf4_score(keys, responds):
    scores = np.zeros((len(keys), len(responds)))
    for i in range(len(keys)):
        for j in range(len(responds)):
            scores[i, j] = phi4(keys[i], responds[j])

    matching = linear_assignment(-scores)
    similarity = float(sum(scores[matching[:, 0], matching[:, 1]]))

    p = similarity / len(responds) if similarity else 0.0
    r = similarity / len(keys) if similarity else 0.0

    return p, r, f1_score(p, r)

def total_num_links(gold_clusters, auto_clusters):
    gold_ms = {m for gc in gold_clusters for m in gc}
    auto_ms = {m for ac in auto_clusters for m in ac}
    num_ms = len(gold_ms.union(auto_ms))
    num_links = (num_ms * (num_ms - 1)) / 2

    return num_links

def Blanc_score(keys, responds):

    def get_links(cluster):
        cluster = list(cluster)
        if len(cluster) > 1:
            links = {(m1, m2) if m1 < m2 else (m2, m1) for i, m1 in enumerate(cluster) for m2 in cluster[i + 1:]}
            # print(links)
            return links
            # return set(itertools.combinations(cluster, 2))
        else:
            return set()

    gold_links = set.union(*map(get_links, keys))
    auto_links = set.union(*map(get_links, responds))

    num_links = total_num_links(keys, responds)

    # coreferent / non-coreferent indices
    c, n = 0, 1
    confusion = np.zeros((2, 2), dtype="int32")

    confusion[c, c] = len(auto_links & gold_links)  # intersection of links
    confusion[n, c] = len(auto_links.difference(gold_links))  # (auto union gold) \ gold
    confusion[c, n] = len(gold_links.difference(auto_links))  # (auto union gold) \ auto
    confusion[n, n] = num_links - (confusion[c, c] + confusion[n, c] + confusion[c, n])

    pc = float(confusion[c, c]) / (confusion[c, c] + confusion[n, c]) \
        if confusion[c, c] + confusion[n, c] > 0 \
        else 0.0
    pn = float(confusion[n, n]) / (confusion[c, n] + confusion[n, n]) \
        if confusion[c, n] + confusion[n, n] > 0 \
        else 0.0
    p = float(pc + pn) / 2

    rc = float(confusion[c, c]) / (confusion[c, c] + confusion[c, n]) \
        if confusion[c, c] + confusion[c, n] > 0 \
        else 0.0
    rn = float(confusion[n, n]) / (confusion[n, c] + confusion[n, n]) \
        if confusion[n, c] + confusion[n, n] > 0 \
        else 0.0
    r = float(rc + rn) / 2

    fc = f1_score(pc, rc)
    fn = f1_score(pn, rn)
    f = float(fc + fn) / 2

    return p, r, f


def links2clusters_new(triplets):
    clusters = []
    mention2clusters = {}

    current_mention = ""
    prev_mentions = []
    prev_scores = []



    for triplet in triplets:
        if triplet[0] not in mention2clusters:
            mention2clusters[triplet[0]] = len(clusters)
            clusters.append({triplet[0]})

        if triplet[1] != current_mention:
            # save the previous state, make decision on where to put the mention
            if len(prev_scores) > 0:
                maximum_score = max(prev_scores)
                if maximum_score < 0.5:
                    # This mention should be in a new cluster
                    mention2clusters[current_mention] = len(clusters)
                    clusters.append({current_mention})
                else:
                    best_fit_mention = prev_mentions[np.argmax(prev_scores)]
                    cluster_id = mention2clusters[best_fit_mention]
                    clusters[cluster_id].add(current_mention)
                    mention2clusters[current_mention] = cluster_id

            # start new classifying mention
            current_mention = triplet[1]
            prev_mentions = []
            prev_scores = []

        prev_mentions.append(triplet[0])
        prev_scores.append(triplet[2])

    # save the final dict state, make decision on where to put the mention
    return clusters


if __name__ == "__main__":

    with open("../data/data_set_keys_scene_tst.json") as fin:
        mention_list = json.load(fin)

    model_logits = np.load("2020-06-26-22-48-54_tst_logits.npy")

    A = [{1, 2, 3, 4, 5}, {6, 7}, {8, 9, "a", "b", "c"}]
    B = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, "a", "b", "c"}]

    print(MUC_score(A, B))
    print(Bcube_score(A, B))
    print(Ceaf4_score(A, B))

    A = [{1, 2, 3, 4, 5}, {6, 7}, {8, 9, "a", "b", "c"}]
    B = [{1, 2, 3, 4, 5, 8, 9, "a", "b", "c"}, {6, 7}]
    print(MUC_score(A, B))
    print(Bcube_score(A, B))
    print(Ceaf4_score(A, B))

    # All test passed

    # Test real random cases
    gold_dev_cluster = links2clusters(mention_list)
    all_logits = np.random.randint(2, size=len(mention_list))
    all_logits = all_logits > 0.5
    random_cluster = score2clusters(mention_list, all_logits)




    # Test_trplets = [["s01_e22_c01_u001_m00001", "s01_e22_c01_u003_m00001", 1], ["s01_e22_c01_u001_m00001", "s01_e22_c01_u003_m00002", 1], ["s01_e22_c01_u003_m00001", "s01_e22_c01_u003_m00002", 1], ["s01_e22_c01_u001_m00001", "s01_e22_c01_u004_m00001", 1], ["s01_e22_c01_u003_m00001", "s01_e22_c01_u004_m00001", 1], ["s01_e22_c01_u003_m00002", "s01_e22_c01_u004_m00001", 1], ["s01_e22_c01_u001_m00001", "s01_e22_c01_u006_m00001", 1], ["s01_e22_c01_u003_m00001", "s01_e22_c01_u006_m00001", 1], ["s01_e22_c01_u003_m00002", "s01_e22_c01_u006_m00001", 1], ["s01_e22_c01_u004_m00001", "s01_e22_c01_u006_m00001", 1], ["s01_e22_c01_u001_m00001", "s01_e22_c01_u007_m00001", 0], ["s01_e22_c01_u003_m00001", "s01_e22_c01_u007_m00001", 0], ["s01_e22_c01_u003_m00002", "s01_e22_c01_u007_m00001", 0], ["s01_e22_c01_u004_m00001", "s01_e22_c01_u007_m00001", 0], ["s01_e22_c01_u006_m00001", "s01_e22_c01_u007_m00001", 0], ["s01_e22_c01_u001_m00001", "s01_e22_c01_u007_m00002", 0], ["s01_e22_c01_u003_m00001", "s01_e22_c01_u007_m00002", 0], ["s01_e22_c01_u003_m00002", "s01_e22_c01_u007_m00002", 0], ["s01_e22_c01_u004_m00001", "s01_e22_c01_u007_m00002", 0], ["s01_e22_c01_u006_m00001", "s01_e22_c01_u007_m00002", 0]]
    cluster_result = links2clusters_new(mention_list)

    model_result = score2clusters_new(mention_list, model_logits)

    print(MUC_score(cluster_result, model_result))
    print(Bcube_score(cluster_result, model_result))
    print(Ceaf4_score(cluster_result, model_result))
    print(Blanc_score(cluster_result, model_result))


    # print(links2clusters_new(mention_list))




