FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

COPY . /home/3D-MRI-style-transfer
RUN pip install -r /home/3D-MRI-style-transfer/requirements.txt

RUN echo "Successfully build image!"

WORKDIR /home/3D-MRI-style-transfer
CMD ["python", "/home/3D-MRI-style-transfer/test.py", "--name", "experiment", "--gpu_ids", "-1", "--dataroot", "/var/dataset", "--CUT_mode", "CUT", "--mean", "88.46", "--std", "32.10", "--model", "cut", "--netG", "resnet", "--netD", "n_layers", "--num_test", "100", "--dataset_mode", "image_ct", "--input_nc", "1", "--output_nc", "1", "--ngl", "4", "--n_downsampling", "1"]