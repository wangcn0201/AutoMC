import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) 
import prune_C1
import prune_C2
import prune_C3
import prune_C4
import prune_C5
import prune_C7


def run(data_name, arch_name, alg, rate):
    data = {'dir': './data', 'name': data_name}
    arch = {'dir': './trained_models/{}/{}.pth.tar'.format(data['name'], arch_name), 'name': arch_name}
    save_dir = './snapshots/{}/{}/{}/'.format(data['name'], arch_name, rate)

    if '1' in alg:
        print("prune_C1:")
        save_dir = '{}C1/'.format(save_dir)
        res = prune_C1.knowledge_distillation(data, save_dir, arch, rate_based_on_original=rate, fixed_seed=True).main()

    elif '2' in alg:
        print("prune_C2:")
        save_dir = '{}C2/'.format(save_dir)
        res = prune_C2.LeGR(data, save_dir, arch, rate=rate, fixed_seed=True).main()

    elif '3' in alg:
        print("prune_C3:")
        save_dir = '{}C3/'.format(save_dir)
        res = prune_C3.NetworkSlimming(data, save_dir, arch, rate=rate, fixed_seed=True).main()

    elif '4' in alg:
        print("prune_C4:")
        save_dir = '{}C4/'.format(save_dir)
        res = prune_C4.SoftFilterPruning(data, save_dir, arch, rate=rate, fixed_seed=True).main()

    elif '5' in alg:
        print("prune_C5:")
        save_dir = '{}C5/'.format(save_dir)
        res = prune_C5.HOS(data, save_dir, arch, rate=rate, fixed_seed=True).main()

    elif '7' in alg:
        print("prune_C7:")
        save_dir = '{}C7/'.format(save_dir)
        res = prune_C7.LFB(data, save_dir, arch_name, rate=rate, fixed_seed=True).main()
    
    print(res)


if __name__ == '__main__':
    data_name, arch_name, alg, rate = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    run(data_name, arch_name, alg, float(rate))
    