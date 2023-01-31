from utils import img_util
import prepare_dataset as prep


if __name__=="__main__":
    val_dir = '/home/ubuntu/resources/shared_efs/vrdu/datasets/docvqa/test/'
    val_res = prep.get_img2doc_data(val_dir + 'documents')

    