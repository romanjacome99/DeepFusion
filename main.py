import tensorflow as tf
import os

from unrolling_network import *
from custom_loss_metrics import *
from callbacks import *
from read_datasets import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
            # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
        print(e)


#---------- Params -----------------------
M = 256
L = 31
ds = 0.5
dl = 0.5
batch_size = 1
stages_t = 10
snr = 30
noise = True
reg_param = 0.5
sig_param = 10
opt_H = True

# Network configuration
config_loss = 'ML_T'
config_t = 'T_reg_s'  # ['NT', 'T_reg_n', 'T_reg_s']

print('Training model: Loss: ' + config_loss + ' | CA training: ' + config_t)
if config_t == 'T_reg_n':
    type_reg = 'Normal'
    in_reg_param = 0.5
else:
    type_reg = 'sig'
    in_reg_param = 30


dataset_dict = 'rad'

train_path = "/home/jorgehdsp/Documents/Roman_code/ICVL/Train"
val_path = "/home/jorgehdsp/Documents/Roman_code/ICVL/validation"
test_path = "/home/jorgehdsp/Documents/Roman_code/ICVL/Test"


# Callbacks
[train_gen, val_gen, test_gen] = generate_dataset(batch_size, M, L, train_path, val_path, test_path, dataset_dict)
experiment = "results_mean_loss"
# Callbacks
path = "results/"+experiment+"/"
csv_file = path + experiment+".csv"

model_path = path +experiment+".h5"

try:
    os.mkdir(path)
except OSError as error:
    print(error)

save_model_callback_all = save_each_epoch(model_path)

callbacks = [save_model_callback_all,
             tf.keras.callbacks.CSVLogger(csv_file, separator=',', append=False)]

lr = 0.001

optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                     name='Adam')



model = unrollingNetwork(input_dim=(M, M, L), HO=(0.25, 0.5, 0.25), ds=ds, stage_t=stages_t, shots=1, batch_size=batch_size, dl=dl,noise=noise,snr=snr,opt_H=opt_H,reg_param=reg_param,sig_param=sig_param,type_reg=type_reg)

rho_l = 1
rho_s = 1
model.compile(optimizer=optimizer, loss=loss_unrolling(rho_l=rho_l,rho_s=rho_s), metrics=psnr_metric)

# model.summary()

h = model.fit(x=train_gen, validation_data=val_gen,verbose=1, epochs=300, callbacks=[callbacks])