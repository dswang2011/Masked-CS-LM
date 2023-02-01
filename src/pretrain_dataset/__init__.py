

def setup(opt):
    if opt.pretrain_dataset.lower() == 'rvlcdip':
        from pretrain_dataset.rvl_cdip import RVLCDIP as MyData
    else:
        raise Exception('dataset not supported:{}'.format(opt.dataset_name))

    mydataset = MyData(opt=opt)
    return mydataset
