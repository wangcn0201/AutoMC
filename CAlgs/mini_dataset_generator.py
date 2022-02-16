import os
import random
import pickle
import torch
from utils import *
from models import *

class mini_dataset:
    def __init__(self, original_data, save_dir, arch, rate):
        self.original_data_dir = original_data['dir']
        self.original_data_name = original_data['name']
        self.save_dir = save_dir
        self.arch = arch['dir']
        self.arch_name = arch['name']
        self.model = torch.load(self.arch)
        self.num_classes = get_num_classes(self.original_data_name)
        self.rate = rate
        if '100' in original_data['name']:
            self.num_classes = 100
        else: self.num_classes = 10

    def val(self, model, data_loader):
        model.eval()
        correct_index, wrong_index = {}, {}
        for i in range(self.num_classes):
            correct_index[i], wrong_index[i] = [], []
        with tqdm(total=len(data_loader)) as t:
            for index, (input, label) in enumerate(data_loader):
                input, label = input.cuda(), label.cuda()
                input, label = Variable(input), Variable(label)

                output = model(input)

                _, pred = output.topk(1, 1, True, True)
                pred = pred.t()
                result = bool(pred.eq(label.view(1, -1).expand_as(pred))[0][0])

                if result: correct_index[int(label[0])].append(index)
                else: wrong_index[int(label[0])].append(index)

                t.update()

        return correct_index, wrong_index

    def select_data_index(self, correct_index, wrong_index, rate=0.1):
        correct_num, wrong_num = len(correct_index), len(wrong_index)
        all_num = (correct_num + wrong_num)
        selected_all_num = int(all_num * rate)
        selected_correct_num = int(correct_num * rate)
        selected_wrong_num = selected_all_num - selected_correct_num
        assert selected_wrong_num <= wrong_num

        selected_correct_index = random.sample(correct_index, selected_correct_num)
        selected_wrong_index = random.sample(wrong_index, selected_wrong_num)

        return selected_correct_index, selected_wrong_index

    def get_mini_dataset(self, dataset_type, data, loader):
        print(dataset_type + ': ')
        correct_path, wrong_path = '{}{}_correct_index.npy'.format(self.save_dir, dataset_type), '{}{}_wrong_index.npy'.format(self.save_dir, dataset_type)
        if os.path.exists(correct_path):
            correct_index = np.load(correct_path, allow_pickle=True).item()
            wrong_index = np.load(wrong_path, allow_pickle=True).item()
        else:
            correct_index, wrong_index = self.val(self.model, loader)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            np.save(correct_path, correct_index)
            np.save(wrong_path, wrong_index)

        correct_num, wrong_num = 0, 0
        selected_index = []

        for i in range(self.num_classes):
            correct_len = len(correct_index[i])
            wrong_len = len(wrong_index[i])

            correct_num += correct_len
            wrong_num += wrong_len

            print(correct_len, wrong_len)
            sci, swi = self.select_data_index(correct_index[i], wrong_index[i], rate=self.rate)
            selected_index += sci + swi

        print(correct_num / (correct_num + wrong_num))

        mini_data = torch.utils.data.Subset(data, selected_index)
        f_data = open(self.save_dir + 'mini_{}_data'.format(dataset_type), 'wb')
        pickle.dump(mini_data, f_data)
        f_data.close()

    def main(self):
        train_data, val_data, train_loader, val_loader = load_data(data_name=self.original_data_name, data_dir=self.original_data_dir, batch_size=1, return_data=True)
        self.get_mini_dataset('train', train_data, train_loader)
        self.get_mini_dataset('val', val_data, val_loader)
        mini_train_loader, mini_val_loader = load_mini_data(self.save_dir)
        validate(self.model, mini_train_loader)
        validate(self.model, mini_val_loader)
        
'''
if __name__ == '__main__':
    arch_name = 'vgg11'
    arch = {'dir': './trained_models/{}.pth.tar'.format(arch_name), 'name': arch_name}
    data = {'dir': './data', 'name': 'cifar10'}
    rate = 0.1
    save_dir = './data/mini_dataset/{}/{}/{}/'.format(data['name'], arch_name, rate)
    mini_dataset(data, save_dir, arch, rate).main()
'''