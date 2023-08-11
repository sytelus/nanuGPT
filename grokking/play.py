
def load_deb(filename):
    deb = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                line = line.split()
                deb.append(tuple(int(i) for i in line))
    return deb

def sort_deb(deb):
    # Sort by each column
    for i in range(len(deb[0])):
        deb = sorted(deb, key=lambda x: x[i])
    return deb

deb_high = load_deb("all_train_high.txt")
deb_low = load_deb("all_train_low.txt")

# deb_high = sort_deb(deb_high)
# deb_low = sort_deb(deb_low)

# assert(len(deb_high)==len(deb_low))

# deb_high_set = set(deb_high)
# deb_low_set = set(deb_low)

# assert(len(deb_high_set)==len(deb_low_set))

# intersect = set.intersection(deb_high_set, deb_low_set)
# print('intersect', len(intersect), len(deb_high_set), len(deb_low_set))

deb_high_dict = {}
for i in deb_high:
    deb_high_dict[i] = deb_high_dict.get(i, 0) + 1

deb_low_dict = {}
for i in deb_low:
    deb_low_dict[i] = deb_low_dict.get(i, 0) + 1

diff=0
for k, i in deb_high_dict.items():
    diff += abs(deb_low_dict[k] - i)
print('diff', diff)
print('len', len(deb_high_dict), len(deb_low_dict))
print('done')
