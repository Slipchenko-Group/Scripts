import calculate_rmsd as rmsd, numpy as np

def min_dist_index(dist):
    min_index = 0
    min_dist = dist[0][2]
    for i in range(1, len(dist)):
        if dist[i][2] < min_dist:
            min_dist = dist[i][2]
            min_index = i

    return min_index


def order(results, num):
    ret = []
    for i in range(num):
        for j in range(len(results)):
            if results[j][0] == i:
                ret.append(results[j])

    return ret


def arrange(ref, to):
    ref -= rmsd.centroid(ref)
    to -= rmsd.centroid(to)
    U = rmsd.kabsch(to[:3, :], ref[:3, :])
    to = np.dot(to, U)
    dist = []
    ref_CT_taken = []
    to_CT_taken = []
    unordered_results = []
    for i in range(3, len(ref)):
        for j in range(3, len(to)):
            ref_CT = ref[i, :]
            to_CT = to[j, :]
            dist.append([i - 3, j - 3, abs(np.linalg.norm(ref_CT - to_CT))])

    while len(ref_CT_taken) != len(ref) - 3:
        min_index = min_dist_index(dist)
        unordered_results.append([dist[min_index][0], dist[min_index][1]])
        ref_lmo = dist[min_index][0]
        to_lmo = dist[min_index][1]
        ref_CT_taken.append(ref_lmo)
        to_CT_taken.append(to_lmo)
        to_be_deleted = []
        for i in range(len(dist)):
            if dist[i][0] == ref_lmo:
                if i not in to_be_deleted:
                    to_be_deleted.append(dist[i])
                continue
            if dist[i][1] == to_lmo:
                if i not in to_be_deleted:
                    to_be_deleted.append(dist[i])

        for i in range(len(to_be_deleted)):
            dist.remove(to_be_deleted[i])

    return order(unordered_results, len(ref) - 3)

