
from deep_prior_networks import *
from unrolling_layers import *
from sensing_layers import *
from tensorflow.keras.models import Model

def recursion(X, y_m, H_m, y_c, H_c, u, input_dim=(512, 512, 31), HO=(0.25, 0.5, 0.25), ds=0.5, stage=1, shots=1, batch_size=1, dl=0.5):
    
    Xc = GradientCASSI(input_dim=input_dim, HO=HO, ds=ds, name=False, shots=shots, batch_size=batch_size)([X,y_c,H_c])
    Xc = Conv2DTranspose(input_dim[-1],(3,3),strides=(int(1/ds),int(1/ds)),padding='same')(Xc)

    Xm = GradeientMCFA(input_dim=input_dim,dl=dl,shots=shots,batch_size=batch_size)([X,y_m,H_m])
    Xm = Conv2D(input_dim[-1],(1,1),padding='same')(Xm)

    h = prior_highres(X-u,Bands=input_dim[-1],Kernels_Size=(3,3),num_filters=20,trainable=True)

    reg_term = AdaptiveParameter(name = 'mu_'+str(stage))(X - (h + u))

    Gd = AdaptiveParameter(name='lambda_'+str(stage))(reg_term+Xc+Xm)

    Xk = Subtract(name='X'+str(stage))([X,Gd])

    dual = AdaptiveParameter(name='alpha_'+str(stage))(h-Xk)



    uk = Add(name='u'+str(stage))([u,dual])

    return Xk,uk

def unrollingNetwork(input_dim=(512, 512, 31), HO=(0.25, 0.5, 0.25), ds=0.5, stage_t=10, shots=1, batch_size=1, dl=0.5,noise=True,snr=30,opt_H=True,reg_param=0.5,sig_param=10,type_reg='sig'):
    GT = Input(shape=input_dim)

    
    # Sensing simulations

    [Xm, ym, Hm] = ForwardMCFA(input_dim=input_dim, noise=noise, reg_param=reg_param, dl=dl, opt_H=opt_H,
                 name='MCFA_Layer', type_reg=type_reg, sig_param=sig_param, shots=shots, batch_size=batch_size,snr=snr)(GT)

    [Xc, yc, Hc] = ForwardCASSI(input_dim=input_dim, noise=noise, reg_param=reg_param, ds=ds, opt_H=opt_H, HO=HO,
                 name='CASSI_Layer', type_reg=type_reg, sig_param=sig_param, shots=shots, batch_size=batch_size,snr=snr)(GT)
    

    Xc = Conv2DTranspose(input_dim[-1],(3,3),strides=(int(1/ds),int(1/ds)),padding='same')(Xc)

    Xm = Conv2D(input_dim[-1],(1,1),padding='same')(Xm)

    Xk = Add()([Xc,Xm])


    X = []
    X.append(Xk)
    uk = tf.zeros([batch_size,input_dim[0],input_dim[1],input_dim[2]])
    for stage in range(stage_t):
        [Xk,uk] = recursion(Xk, ym, Hm, yc, Hc, uk, input_dim=input_dim, HO=HO, ds=ds, stage=stage, shots=shots, batch_size=batch_size, dl=dl)
        X.append(Xk)
    
    model = Model(GT,X)
    return model

    
