import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import train


def t(data_name, arch_name, epochs):
    data = {'dir':'./data', 'name':data_name}
    path = './snapshots/{}/{}/train/'.format(data_name, arch_name)
    print(train.Train(data, path, arch_name, epochs=epochs).main())

if __name__ == '__main__':
    data_name, arch_name, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    t(data_name, arch_name, epochs)