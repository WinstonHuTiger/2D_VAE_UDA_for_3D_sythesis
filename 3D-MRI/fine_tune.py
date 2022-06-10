import time
import torch
from tqdm import tqdm
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.ssim import SSIM
from util.visualizer import Visualizer
import numpy as np


if __name__ == '__main__':
   
    opt = TrainOptions().parse()   # get training options
    assert (opt.dataset_mode== "brain_3D_transfer") # assert the dataset mode is brain_3D_transfer
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    opt.dataset_size = dataset_size
    opt.print_freq = max(1, dataset_size // opt.batch_size // opt.print_freq)
    opt.display_freq = max(1, dataset_size // opt.batch_size // opt.display_freq)

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations
    validation_loss_fun = torch.nn.L1Loss()
    validation_loss_fun2 = SSIM()

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        pbar = tqdm(dataset, total=int(len(dataset) / opt.batch_size))
        message = '(epoch: %d)'%epoch
        for i, data in enumerate(pbar):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            dataset.dataset.updateDataAugmentation()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data[0]["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data[0])
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data[0])  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            del data[0]
            # fine tuning based on vae encoder
            # fake_T1 (domain2) <--> ground truth T1 (domain2)
            if opt.netVae_path is not None:
                model.set_input(data[-1])
                model.optimize_parameters_kl()
            else:
                # fine tuning based on adverserial loss 
                # fake_T1 (domain2) <--> ground truth T1 (domain2)
                model.set_input(data[-1])
                model.optimize_parameters_wo_l1()
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time
            
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = epoch % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
               
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                message = visualizer.print_and_get_loss_message(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            torch.cuda.empty_cache()

        # validation_loss_array = []
        # validation_loss_array2 = []
        # opt.phase='val'
        # tmp = opt.serial_batches, opt.paired
        # opt.serial_batches=True
        # opt.paired = True
        # for i in tqdm(range(int(len(test_dataset) / opt.batch_size)), desc='(epoch %d) Validation'%epoch):
        #     test_data = next(test_dataset_iter, None)
        #     if test_data is None:
        #         test_dataset_iter = iter(test_dataset)
        #         test_data = next(test_dataset_iter, None)

        #     model.set_input(test_data)
        #     model.test()
        #     validation_loss_array.append(validation_loss_fun(model.fake_B, model.real_B).item())
        #     validation_loss_array2.append(1 - validation_loss_fun2(model.fake_B, model.real_B).item())
        
        # val_loss = {'L1': np.mean(validation_loss_array), '1-SSIM': np.mean(validation_loss_array2)}
        # visualizer.print_validation_loss(epoch, val_loss)
        # visualizer.plot_current_validation_losses(epoch, val_loss)
        # opt.phase='train'
        # opt.serial_batches, opt.paired = tmp

        model.save_networks('latest')
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
