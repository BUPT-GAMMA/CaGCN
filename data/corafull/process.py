import numpy as np
def main(labelrate):
    with open('train%s.txt' % labelrate, 'r') as f:
        train_index = f.read().splitlines()
        train_index = list(map(int, train_index))
    with open('test%s.txt' % labelrate, 'r') as f:
        test_index = f.read().splitlines()
        test_index = list(map(int, test_index))
    index = []
    for i in range(0, 19792):
        if i in train_index or i in test_index:
            continue
        else:
            index.append(i)
    index = np.array(index)
    randlist = np.random.randint(low=0, high=len(index), size=500)
    val_index = index[randlist]
    with open('val%s.txt'%labelrate, 'w') as f:
        for i in val_index:
            f.write(str(i))
            f.write('\n')

if __name__ == '__main__':
    main(60)

