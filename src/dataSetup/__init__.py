def setup(opt):
    if opt.dataset_name.lower() == 'funsd':
        from dataSetup.funsd import FUNSD as MyData
    else:
        raise Exception('dataset not supported:{}'.format(opt.dataset_name))

    mydataset = MyData(opt=opt)
    return mydataset
