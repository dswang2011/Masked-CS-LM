

def setup(opt):
    if opt.pretrain_dataset.lower() == 'funsd4g':
        from dataSetup.funsd4g import FUNSD as MyData
    else:
        raise Exception('dataset not supported:{}'.format(opt.dataset_name))

    mydataset = MyData(opt=opt)
    return mydataset
