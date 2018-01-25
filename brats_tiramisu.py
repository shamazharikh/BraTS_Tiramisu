from tiramisu import Tiramisu
from bratsdata import BratsData, BratsValidationData
import os


if __name__ == '__main__':
    batch_size = 1

    dataobj = BratsData(
        parent_dir='../Brats17TrainingData/',
        test_split=0.1,
        validation_split=0.1,
        lgg=True,
        hgg=True,
        tumour=True,
        batch_size=batch_size)
    # dataobj = BratsValidationData(
    #     parent_dir='./Brats17ValidationData/HGG/',
    #     batch_size=batch_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    u = Tiramisu(output_filters=1,
        input_width=240,
        input_height=240,
        input_channels=4,
        n_classes=3,
        growth=16,
        #class_weights=[ 0.02067452,  0.0727702 ,  2.30224412,  1.        ,  3.23101353],
        class_weights=[1,1,1]
        best_score={'WT':0.00,'TC':0.00,'AT':0.00},
        saved='saved/residual_tiramisu/checkpoints/residual_tiramisu_{}',
        layer_depths=[4,5,7,10,12],
        long_residual=True,
        short_residual=False,
        bottle_neck=15,
        memory_fraction=0.4,
        batch_size=batch_size)

    # u.predict(dataobj.get_data)

    # u.train_net(
    #    dataobj.get_train_data,
    #    dataobj.get_validation_data,
    #    restore=False,
    #    n_epochs=100)
    
    u.reinforce(
        dataobj.get_train_data,
        dataobj.get_validation_data,
        restore=False,
        n_epochs=100)
