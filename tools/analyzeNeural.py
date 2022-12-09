import numpy as np
import matplotlib.pyplot as plt
import keras
from tqdm import tqdm
from sklearn.decomposition import PCA
'''
##################################
TODO:
Put in synaptic masking function into analysis functions

#################################
'''



"""#######################
Response spectrum analysis
#######################"""
def spectralResponse(model,test_spectrum= np.linspace(.1,15,50),batch_size=8,
                    t = np.arange(-1,40,1/120),resolution=1,n_samp=None,figure=None,
                    peptides=1,return_all=False,**kwargs,
                    ):
    if t.max()<test_spectrum.max():
        print('WARNING: simulated time range does not capture full period of all stimuli')
    from .model import get_n_neuron
    n_neuron = get_n_neuron(model)
    #generate simulated data
    U_test = []
    Z_test = []
    t = np.arange(-1,40,1/120)
    for i in test_spectrum:
        U_test.append(np.sin(t/i*2*np.pi)/2+.5)
        Z_test.append(np.sin(t/i*2*np.pi)/2+.5)
    U_test = np.array(U_test)[...,None]
    UU_gene = np.ones((U_test.shape[0],1))*peptides
    UU_ablate = np.ones((U_test.shape[0],n_neuron))
    UU_syn = np.ones((U_test.shape[0],n_neuron))
    spec_pred = model.predict([U_test,UU_gene,UU_ablate,
                    UU_syn[:test_spectrum.size]],batch_size=batch_size,verbose=1)
    #calculate correlations
    def cross_correlate(x,U,tau):
        tau = tau.astype(int)
        if len(x.shape)>1:
            xx = np.median(x,axis=0)
        else:
            xx=x.copy()
        return np.array([np.cov(xx[t:],U[:-t])[0,1] for t in tau])
    amp = []
    phase=[]
    for i,(z,u) in tqdm(enumerate(zip(spec_pred,Z_test))):
        if test_spectrum[i]*120<1: #an exception needed for very fast periods
            amp.append(np.nan)
            phase.append(np.nan)
            continue
        tau = np.arange(1,test_spectrum[i]*120)[::resolution]
        if n_samp is None:
            n_samp = int(test_spectrum.max()*120)
            # n_samp = min(2*int(test_spectrum[i]*120),n_samp)
            # n_samp = max(n_samp,10)
        corr=cross_correlate(z[-n_samp:],u[-n_samp:],tau)
        amp.append(np.nanmax(corr))
        phase.append(np.argmax(corr)/corr.size)
    # center phase range on zero
    phase = np.array(phase)
    phase[phase>.5] -= 1
    #plot it
    fig,ax = plot_spectrum(test_spectrum,amp,phase,figure,**kwargs)
    if return_all:
        return fig, ax, test_spectrum, amp, phase, spec_pred,U_test
    return fig,ax

def spectralResponse_ablation(model,test_spectrum= np.linspace(.1,15,50),ablation=np.linspace(0,.5,5),n_ablate=24,batch_size=8,
                    t = np.arange(-1,40,1/120),resolution=1,n_samp=None,figure=None,
                    peptides=1,return_all=False,**kwargs,
                    ):
    if t.max()<test_spectrum.max():
        print('WARNING: simulated time range does not capture full period of all stimuli')
    from .model import get_n_neuron, ablate_square
    n_neuron = get_n_neuron(model)
    batch_size = min(batch_size,n_ablate)
    print('ablation',ablation)
    amp = []
    phase=[]
    #Loop through ablations
    for j,abl in enumerate(ablation):
        amp_ablate = []
        phase_ablate = []
        #test each frequency
        for i,period in enumerate(test_spectrum):
            #generate simulated data
            t = np.arange(-1,40,1/120)
            U_test = np.sin(t/i*2*np.pi)/2+.5
            U_test = np.array([U_test for _ in range(n_ablate)])[...,None]
            UU_gene = np.ones((U_test.shape[0],1))*peptides
            UU_ablate = ablate_square(abl,n_neuron,n_ablate)
            UU_syn = np.ones((U_test.shape[0],n_neuron))
            spec_pred = model.predict([U_test,UU_gene,UU_ablate,UU_syn],
                                    batch_size=batch_size,verbose=1)
            #calculate correlations
            def cross_correlate(x,U,tau):
                tau = tau.astype(int)
                if len(x.shape)>1:
                    xx = np.median(x,axis=0)
                else:
                    xx=x.copy()
                return np.array([np.cov(xx[t:],U[:-t])[0,1] for t in tau])
            #make list for amplitudes and phases at this ablation level
            amp_ablate.append([])
            phase_ablate.append([])
            for ii,z in tqdm(enumerate(spec_pred)):
                if test_spectrum[i]*120<1: #an exception needed for very fast periods
                    amp.append(np.nan)
                    phase.append(np.nan)
                    continue
                tau = np.arange(1,test_spectrum[i]*120)[::resolution]
                if n_samp is None:
                    n_samp = int(test_spectrum.max()*120)
                    # n_samp = min(2*int(test_spectrum[i]*120),n_samp)
                    # n_samp = max(n_samp,10)
                corr=cross_correlate(z[-n_samp:],U_test[0,-n_samp:,0],tau)
                amp_ablate[-1].append(np.nanmax(corr))
                phase_ablate[-1].append(np.argmax(corr)/corr.size)
            #end period for this ablation

        #plot for this ablation level
        # center phase range on zero
        phase_ablate = np.array(phase_ablate)
        phase_ablate[phase_ablate>.5] -= 1
        amp_ablate=np.array(amp_ablate)
        #plot it
        c=plt.cm.viridis(j/len(ablation))
        figure = plot_spectrum(test_spectrum,np.median(amp_ablate,axis=1),np.median(phase_ablate,axis=1),
                            figure,color=c,line=True,**kwargs,)
        #stack on data
        phase.append(phase_ablate)
        amp.append(amp_ablate)
        #end(this ablation level)
    fig,ax=figure
    if return_all:
        return fig, ax, test_spectrum, amp, phase, spec_pred,U_test
    return fig,ax

def plot_spectrum(spectrum,amp,phase,figure=None,label=None,color=None,line=False):
    if figure is None:
        fig,ax = plt.subplots(nrows=2,figsize=(6,8),sharex=True)
    else:
        fig, ax = figure
    if line:
        ax[0].plot(spectrum,amp,color=color,label=label)
    else:
        ax[0].scatter(spectrum,amp,color=color,label=label)
    ax[0].set_ylabel('signal covariance')

    if line:
        ax[1].plot(spectrum,phase,color=color,label=label)
    else:
        ax[1].scatter(spectrum,phase,color=color,label=label)
    ax[1].set_xlabel('1/f (min)')
    ax[1].set_ylabel('phase angle ($2\pi $ rad)')
    plt.legend()
    return fig, ax

"""#######################
Noise Autocovariance analysis
#######################"""
def noise_autocovariance(model, n_test=100, t_test=60, peptides=1,batch_size=8,
                        tau=np.arange(1,120*15,3),figure=None,**kwargs):
    from .model import get_n_neuron
    n_neuron = get_n_neuron(model)
    #generate sample
    U_noise = np.random.normal(.5,.3,(n_test,t_test*120,1))
    Z_noise = model.predict([U_noise,np.ones((n_test,1))*peptides,np.ones((n_test,n_neuron)),
                        np.ones((n_test,n_neuron))],batch_size=batch_size,verbose=1)
    Z_noise = Z_noise[:,30*120:]
    #calculate covariance
    def cross_correlate_auto(x,tau):
        return np.array([np.cov(x[t:],x[:-t])[0,1] for t in tau])
    c = np.array([cross_correlate_auto(z[-1500:],tau) for z in tqdm(Z_noise)])
    #plot
    if figure is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig,ax = figure
    ax.plot(tau/120,np.mean(c,axis=0),**kwargs)
    ax.set_xlabel('$\\tau$ (min)')
    ax.set_ylabel('autocovariance')
    ax.legend()
    return fig,ax

def noise_autocovariance_ablation(model, ablation=np.linspace(0,.75,5), n_test=100,
                        n_ablate=32, t_test=60, peptides=1,batch_size=8,
                        tau=np.arange(1,120*15,3),figure=None,transient=30,**kwargs):
    from .model import get_n_neuron,ablate_square
    n_neuron = get_n_neuron(model)
    for i,abl in enumerate(ablation):
        #generate sample
        U_noise = np.random.normal(.5,.3,(n_test,t_test*120,1))
        U_ablate = ablate_square(abl,n_neuron,n_test) #Note what we're averaging across here SB -120622
        Z_noise = model.predict([U_noise,np.ones((n_test,1))*peptides,U_ablate,
                            np.ones((n_test,n_neuron))],batch_size=batch_size,verbose=1)
        Z_noise = Z_noise[:,transient*120:]
        #calculate covariance
        def cross_correlate_auto(x,tau):
            return np.array([np.cov(x[t:],x[:-t])[0,1] for t in tau])
        c = np.array([cross_correlate_auto(z[:],tau) for z in Z_noise])
        #plot
        if figure is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            fig,ax = figure
        color = plt.cm.viridis(i/len(ablation))
        ax.plot(tau/120,np.mean(c,axis=0),c=color,**kwargs)
        ax.set_xlabel('$\\tau$ (min)')
        ax.set_ylabel('autocovariance')
        ax.legend()
        figure=fig,ax
    return fig,ax


"""#######################
Latent Dynamics analysis
#######################"""

def dynamics_slider(latent_model, x=None, periods=[3,5,10,15], peptides=1, batch_size=8,
                    t = np.arange(-3,40,1/120)):
    ''' makes display slider of neural dynamics. generates data if not provided'''
    from .model import get_n_neuron
    n_neuron = get_n_neuron(latent_model,latent_model=True)
    #generate data
    if x is None:
        U_test = []
        for i in periods:
            U_test.append(np.sin(t/i*2*np.pi)/2+.5)
        U_test = np.array(U_test)[...,None]
        UU_gene = np.ones((U_test.shape[0],1))*peptides
        UU_ablate = np.ones((U_test.shape[0],n_neuron))
        UU_syn = np.ones((U_test.shape[0],n_neuron))
        x = latent_model.predict([U_test,UU_gene,UU_ablate,UU_syn],
                                batch_size=batch_size,verbose=1)

    #make the slider figure
    method=None #normalization option for firing rates, decide whether to keep
    # %matplotlib qt
    from matplotlib.widgets import Slider
    sq = int(n_neuron**.5)
    plot_mat = np.reshape(x,(x.shape[0],x.shape[1],sq,sq))

    if len(plot_mat<=5):
        fig, ax = plt.subplots(ncols=len(plot_mat))#//2,nrows=2)
    else:
        rows = len(plot_mat)//5
        fig, ax = plt.subplots(nrows=len(plot_mat)//5,ncols=5)
    if len(plot_mat)==1:
        ax=[ax]
    else:
        ax = np.ravel(ax)
    fig.suptitle('neural state')
    l=[]
    ind0 = 1
    for i,(a,mat) in enumerate(zip(ax,plot_mat)):
    #    l.append(a.imshow(mat[ind0]/(mat[ind0].max()+1e-10),clim=(0,1)))
    #    l.append(a.imshow(mat[ind0]/(mat[ind0].sum())),)
        if method=='none' or method is None:
            l.append(a.imshow(mat[ind0],clim=(mat.min(),mat.max())))#10)))#
        if method =='mean':
            mtx = mat[ind0]-(mat[ind0].mean())
            l.append(a.imshow(mtx,clim=(-.01,.01),cmap='RdBu'))
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f'{periods[i]}min')
    axidx = plt.axes([0.25, 0.15, 0.65, 0.03])
    slidx = Slider(axidx, 'index', 0, plot_mat[0].shape[0]-1, valinit=ind0, valfmt='%d')
    def update(val):
        idx = slidx.val
        for ll,mat in zip(l,plot_mat):
    #        ll.set_data(mat[int(idx)]/(mat[int(idx)].max())+1e-10)
    #        ll.set_data(mat[int(idx)]/(mat[int(idx)].sum()))
            if method=='none' or method is None:
                ll.set_data(mat[int(idx)])
            if method =='mean':
                mtx=mat[int(idx)]-(mat[int(idx)].mean())+1e-10
                ll.set_data(mtx)
        fig.canvas.draw_idle()
    slidx.on_changed(update)
    plt.show()
    return fig,ax

def firing_pca_fit(latent_model, x=None, periods=[3,5,10,15], peptides=1, batch_size=8,
                    t = np.arange(-3,40,1/120), return_data=False):
    ''' fits a pca object to latent data'''
    from .model import get_n_neuron
    n_neuron = get_n_neuron(latent_model,latent_model=True)
    #generate data
    if x is None:
        U_test = []
        for i in periods:
            U_test.append(np.sin(t/i*2*np.pi)/2+.5)
        U_test = np.array(U_test)[...,None]
        UU_gene = np.ones((U_test.shape[0],1))*peptides
        UU_ablate = np.ones((U_test.shape[0],n_neuron))
        UU_syn = np.ones((U_test.shape[0],n_neuron))
        x = latent_model.predict([U_test,UU_gene,UU_ablate,UU_syn],
                                batch_size=batch_size,verbose=1)
    pca = PCA(n_components=10)
    pca.fit(np.reshape(x[::1],(-1,n_neuron)))
    if return_data:
        return pca, x, U_test
    return pca

def pca_cycles(latent_model,model=None, pca=None, x=None, periods=[3,5,10,15], peptides=1, batch_size=8,
                    t = np.arange(-3,40,1/120), return_data=False, dim=(0,1),
                    figure=None,n_plot=None,syn_gene=0):
    from .model import get_n_neuron
    n_neuron = get_n_neuron(latent_model,latent_model=True)
    #make a pca if not provided
    if pca is None:
        pca,x,U_test = firing_pca_fit(latent_model,x,periods,peptides,batch_size,t,return_data=True)
    elif x is None:
        #generate data
        U_test = []
        for i in periods:
            U_test.append(np.sin(t/i*2*np.pi)/2+.5)
        U_test = np.array(U_test)[...,None]
        UU_gene = np.ones((U_test.shape[0],1))*peptides
        UU_ablate = np.ones((U_test.shape[0],n_neuron))
        # UU_syn = np.ones((U_test.shape[0],n_neuron))
        from .model import synapse_mask_fun
        UU_syn = np.array([synapse_mask_fun(syn_gene,n_neuron) for _ in range(U_test.shape[0])])
        print(UU_syn.mean())
        x = latent_model.predict([U_test,UU_gene,UU_ablate,UU_syn],
                                batch_size=batch_size,verbose=1)

    if n_plot is None:
        n_plot = int(np.max(periods)*120)
    if len(dim)==2:
        if figure is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            fig, ax = figure
        y_all = []
        for i in np.arange(0,x.shape[0],1):
            y = pca.transform(x[i])
            c = plt.cm.viridis(i/x.shape[0])
            plt.plot(y[-n_plot:,dim[0]],y[-n_plot:,dim[1]],c=c)
            y_all.append(y[-n_plot:])
        y_all = np.concatenate(y_all)
        if not model is None:
            y_m = pca.transform(model.layers[-2].get_weights()[0][...,0])/3
            plt.arrow(0,0,y_m[0,dim[0]],y_m[0,dim[1]],head_width=3,zorder=100,color='k',label='motor output')
            plt.scatter([0,0],[0,0],c='k',label='motor output')

            y_in = pca.transform(model.layers[-4].get_weights()[6][None,:])/10
            plt.arrow(0,0,y_in[0,dim[0]],y_in[0,dim[1]],head_width=3,color='firebrick',zorder=100,label='input')
            plt.scatter([0,0],[0,0],color='firebrick',zorder=100,label='input')
        ax.legend(loc=4)
        ax.set_xlabel(f'pca {dim[0]} ({np.round(pca.explained_variance_ratio_[dim[0]],2)})')
        ax.set_ylabel(f'pca {dim[1]} ({np.round(pca.explained_variance_ratio_[dim[1]],2)})')
    #3D case
    else:
        from mpl_toolkits.mplot3d import Axes3D
        if figure is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig,ax = figure
        y_all = []
        for i in np.arange(0,x.shape[0],1):
            y = pca.transform(x[i])
            c = plt.cm.viridis(i/x.shape[0])
            print(n_plot,dim[0])
            ax.plot(y[-n_plot:,dim[0]],y[-n_plot:,dim[1]],y[-n_plot:,dim[2]],c=c)
            y_all.append(y[-n_plot:])
        y_all = np.concatenate(y_all)
        if not model is None:
            y_m = pca.transform(model.layers[-2].get_weights()[0][...,0])/3
            ax.quiver(0,0,0,y_m[0,dim[0]],y_m[0,dim[1]],y_m[0,dim[2]],zorder=100,color='k',)
            ax.scatter([0,],[0,],[0,],c='k',label='motor output')

            y_in = pca.transform(model.layers[-4].get_weights()[6][None,:])/10
            plt.quiver(0,0,0,y_in[0,dim[0]],y_in[0,dim[1]],y_in[0,dim[2]],color='firebrick',zorder=100)
            plt.scatter([0,],[0],[0,],color='firebrick',zorder=100,label='input')
            ax.set_xlabel(f'pca {dim[0]} ({np.round(pca.explained_variance_ratio_[dim[0]],2)})')
            ax.set_ylabel(f'pca {dim[1]} ({np.round(pca.explained_variance_ratio_[dim[1]],2)})')
            ax.set_zlabel(f'pca {dim[2]} ({np.round(pca.explained_variance_ratio_[dim[2]],2)})')
        ax.legend(loc=4)

    return fig, ax
