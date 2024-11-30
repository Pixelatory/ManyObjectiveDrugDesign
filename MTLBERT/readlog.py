import numpy as np

results = []
with open('logs/10-07-2023-09-44-30-smiles_finetune-log.txt', 'r') as f:
    idx = -1
    lines = f.readlines()
    ignore_nums = False
    for line in lines:
        line = line.strip()
        if 'FOLD' in line:
            results.append([])
            idx += 1
            ignore_nums = True
        elif 'AUC' in line or 'R^2' in line or 'Acc' in line or 'RMSE' in line:
            # Only look at AUC and R^2
            results[idx].append([])
            results[idx][-1].append(line)
            ignore_nums = False
        elif not ignore_nums:
            try:
                results[idx][-1].append(float(line))
            except ValueError:
                ignore_nums = True

for prop in range(len(results[0])):
    print(results[0][prop][0])
    print("{:.4f}".format(np.mean([results[fold][prop][-1] for fold in range(len(results))])))

for i in results[-1]:
    print((i[0], i[-1]))

tt = [0 for _ in range(idx+1)]
for fold in range(len(results)):
    for prop in range(len(results[fold])):
        results[fold][prop] = results[fold][prop][-1]

for prop in range(len(results[0])):
    max_idx = [results[i][prop] for i in range(len(results))]
    tt[np.argmax(max_idx)] += 1
'''
if len(tmp) == 5 and ('R^2' in results[i][0] or 'AUC' in results[i][0]):  # 5 is number of folds
    #print(f"{round(float(np.argmin(tmp)), 4):.4f}")
    print(results[i], tmp)
    best_fold.add((results[i][4], tmp[4]))
    tt[np.argmax(tmp)] += 1
'''

print(tt)
exit(1)
lowest_test_loss_idx = np.argmin(results[-1][1:]) + 1
print(lowest_test_loss_idx)
for entry in results:
    print(entry[0], entry[lowest_test_loss_idx], max(entry[1:]), np.argmax(entry[1:]) + 1)
