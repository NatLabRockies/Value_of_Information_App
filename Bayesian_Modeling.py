import math
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.naive_bayes import GaussianNB
import os
import io

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 

def likelihood_KDE(X_train,X_test, y_train, y_test,x_cur, best_parameters):
    """
    Smooth out likelihood function & PLOT using optimal bandwidth and return likelihood (positive & negative)

    Parameters
    ----------
    X_train : array-like [1 x proportion*number samples] (eg 67%)
    X_test : array-like [1 x (1-proportion)*number samples]  (eg 33%)
    x_cur : string, colummn name of attribute/data type being assessed
    best_parameters : dictionary
    Output
    ---------
    pos_like_scaled : (array like) [1 x number of samples]
    neg_like_scaled : (array like) [1 x number of samples]
    x_d (array): 1 x 100,  = np.linspace(min(X_train), max(X_train), 100) 
    count_ij [2 x len(x_d)]
    #
    Can manually fix bandwidth here: bandwidth=1.0 
    otherwise it uses the optimal BANDWIDTH from Naive Bayes grid search
    
    """
    
    kde_pos = KernelDensity(bandwidth=best_parameters['bandwidth'], kernel='gaussian') # best_parameters['bandwidth'] bandwidth=0.3
    kde_neg = KernelDensity(bandwidth=best_parameters['bandwidth'], kernel='gaussian')
    st.write('(in likelihood_KDE) using this otpimized bandwidth:',best_parameters['bandwidth'])
    

    # if np.shape(X_train)[1]>2:
    # if train_test only all features
    # two_d = X_train.iloc[:,x_cur] 
    # x_d = np.linspace(min(X_train.iloc[:, x_cur]), max(X_train.iloc[:, x_cur]), 100) 
    # else:
    # if train_test only gets selected x_cur
    forkde_pos = X_train[y_train>0]#.iloc[:,x_cur] #cur_feat
    forkde_neg = X_train[y_train==0]; 

    # two_d = X_train #.iloc[:,x_cur] #cur_feat
    forkde_pos_np = forkde_pos.values
    forkde_neg_np = forkde_neg.values; 
    kde_pos.fit(forkde_pos_np[:,np.newaxis])
    kde_neg.fit(forkde_neg_np[:,np.newaxis])
    
    # nbins = 100
    x_d = np.arange(np.min(np.concatenate([X_train,X_test])),
                    np.max(np.concatenate([X_train,X_test])),
                    best_parameters['bandwidth']) #np.linspace(min(X_train), max(X_train), nbins) 
    nbins=len(x_d)
    # st.write('min(x_d)',min(x_d),'max(x_d)',max(x_d),'len(x_d)=nbins', nbins)

    Likelihood_logprob_pos = kde_pos.score_samples(x_d[:,np.newaxis]) #.score_samples
    Likelihood_logprob_neg = kde_neg.score_samples(x_d[:,np.newaxis])
    
    #st.write(np.vstack((Likelihood_logprob_pos,Likelihood_logprob_neg)))
    #st.write(np.exp(np.vstack((Likelihood_logprob_pos,Likelihood_logprob_neg))))  

    pos_like_scaled = np.exp(Likelihood_logprob_pos)/np.sum(np.exp(Likelihood_logprob_pos))
    neg_like_scaled = np.exp(Likelihood_logprob_neg)/np.sum(np.exp(Likelihood_logprob_neg))

    #pos_like_scaled = Likelihood_logprob_pos
    #neg_like_scaled = Likelihood_logprob_neg
    # st.write(kde_pos.bandwidth, (X_test.max() - X_test.min()) / kde_pos.bandwidth)

    X_pos_all = pd.concat((X_train[y_train>0],X_test[y_test>0]))
    X_neg_all = pd.concat((X_train[y_train==0],X_test[y_test==0]))

    fig2, ax2 = plt.subplots(figsize=(15,8),ncols=1,nrows=1) # CHANGED to one subplot
    # ax2.hist(X_test,alpha=0.5,color='grey',label='X_test',rwidth=(X_test.max() - X_test.min()) / kde_pos.bandwidth,hatch='/')
    #n_out = ax2.hist([X_test[y_test>0],X_test[y_test==0]], alpha=0.5,facecolor=['g','r'],
    n_out = ax2.hist(X_pos_all, alpha=0.3,facecolor='g',#[X_train[y_train>0]]
                     histtype='bar', hatch='O',edgecolor='grey',label=r'$~Pr(X|\Theta=Positive_{geothermal}$)',bins=x_d) #tacked,bins rwidth= kde_pos.bandwidth) #rwidth= kde_pos.bandwidth,
    n_out = ax2.hist(X_neg_all, alpha=0.3,facecolor='r',#X_train[y_train==0]
                     histtype='barstacked',hatch='/',edgecolor='grey',label=r'$~Pr(X|\Theta=Negative_{geothermal}$)',bins=x_d) #rwidth= kde_pos.bandwidth (X_test.max() - X_test.min()) / 
                     
    ax2.legend(fontsize=18)
    ax2.set_ylabel('Empirical data counts', fontsize=18)
    ax2.tick_params(labelsize=20)
    ax2_ylims = ax2.axes.get_ylim()  

    ax1 = plt.twinx(ax=ax2)
    ax1.fill_between(x_d, pos_like_scaled, alpha=0.3,color='green')
    ax1.plot(x_d,pos_like_scaled,'g.')
    ax1.fill_between(x_d, neg_like_scaled, alpha=0.3,color='red')
    ax1.plot(x_d,neg_like_scaled,'r.')
    ax1.legend(loc=0, fontsize=17)
    ax1.set_ylabel(' Likelihood $~Pr(x | y=Geothermal_{neg/pos}$', fontsize=25)#, rotation=-90)
    ax2.set_xlabel(str(x_cur), fontsize=18)
    ax1.tick_params(labelsize=20)
    ax_ylims = ax1.axes.get_ylim()  
    #print('ax_ylims',ax_ylims)
    #st.write('ax_ylims',ax_ylims)
    ax1.set_ylim(0,ax_ylims[1])
   
    # ax1.set_ylim(0,ax2_ylims[1])
    
    # #.iloc[:,feat4]
    # # n_out = plt.hist([X_test[y_test>0],X_test[y_test==0]], color=['r','g'],histtype='barstacked',rwidth=(X_test.max() - X_test.min()) / kde_pos.bandwidth)
    # #.iloc[:,feat4]
    # n_out = axes[1].hist([X_test[y_test>0],X_test[y_test==0]], color=['g','r'],histtype='barstacked',rwidth=(X_test.max() - X_test.min()) / kde_pos.bandwidth)
    st.pyplot(fig2)
    #st.write('WIDTH of BARS: rwidth=(X_test.max() - X_test.min())',rwidth=(X_test.max() - X_test.min()))    
      
    ### COUNT ARRAY FIGURE # # # # #  #
    #st.write('Staying consistent, rows are *TRUE decision parameter* and columns are *interpretations*.')
    pos_counts = n_out[0][0] 
    neg_counts = n_out[0][1]
    count_ij= np.vstack((n_out[0][0],n_out[0][1]))
    
    #fig3,axes=plt.subplots(nrows=1,ncols=1,figsize=(10,8))
    #axes.imshow(count_ij,vmin=0,vmax=150,cmap='viridis')
    #axes.set_title('Interpretation Counts')

    #for (j,i),label in np.ndenumerate(count_ij):
    #axes.text(i,j,round(label,2),fontsize=18,color='w',ha='center',va='center')

    #xstring = r'''${X}=$'''
    #ystring  = r'''${\Theta}=$'''

    #labels = [item.get_text() for item in axes.get_xticklabels()]
    #empty_string_labels = ['']*len(labels)
    #empty_string_labels[1] = xstring+str(n_out[1][0]);
    #empty_string_labels[3] = xstring+str(n_out[1][int(len(n_out)/2)]);
    #empty_string_labels[5] = xstring+str(n_out[1][-1])
    #axes.set_xticklabels(empty_string_labels)
    #empty_string_labels = ['']*len(labels)
    #empty_string_labels[1] = ystring+'Positive';
    #empty_string_labels[3] = ystring+'Negative';
    #empty_string_labels[5] = ystring+str(+2500)
    #axes.set_yticklabels(empty_string_labels)

    #axes.set_xlabel('Interpretation / Data Attribute ($j$)',fontsize=15)
    #axes.set_ylabel('Pos / Neg Label ($i$)', fontsize=15)
    #st.pyplot(fig3)

    ## RECALCULATE counts with smoothed Likelihood ????
        
       
    #return Likelihood_logprob_pos, Likelihood_logprob_neg, x_d, count_ij 
    # NOT LOG LIKELIHOOD
    return pos_like_scaled, neg_like_scaled, x_d, count_ij 

def Scaledlikelihood_KDE(Pr_prior_POS, Likelihood_neg, Likelihood_pos, X_train,X_test, y_train, y_test,x_cur, x_sampled, best_parameters):
        

    likelihood = np.transpose(np.vstack((Likelihood_neg, Likelihood_pos)))
    #st.write('np.sum(likelihood,1)',np.shape(likelihood),np.sum(likelihood,1))

    X_input_prior_weight_POS = np.outer(np.ones((np.shape(likelihood)[0],)),Pr_prior_POS )
    X_input_prior_weight_NEG = np.outer(np.ones((np.shape(likelihood)[0],)),1.0-Pr_prior_POS )
    X_input_prior_weight= np.hstack((X_input_prior_weight_NEG,X_input_prior_weight_POS))
    ScaledLikelihood = X_input_prior_weight * likelihood

                        
    fig20, ax2 = plt.subplots(figsize=(15,8),ncols=1,nrows=1) # CHANGED to one subplot
    
    n_out = ax2.hist([X_train[y_train>0]], alpha=0.05,facecolor='g',
                    histtype='bar', bins=x_sampled) #tacked,bins rwidth= kde_pos.bandwidth) #rwidth= kde_pos.bandwidth,
    # posi = n_out[0]
    # posi = np.append(posi,0)
    
    n_out = ax2.hist(X_train[y_train==0], alpha=0.05,facecolor='r',
                    histtype='barstacked',bins=x_sampled) #rwidth= kde_pos.bandwidth (X_test.max() - X_test.min()) / 
                    
    
    # ax2.set_ylabel('Empirical data counts', fontsize=18)
    ax2.tick_params(labelsize=20, color='grey')
    ax2_ylims = ax2.axes.get_ylim()  

    ax1 = plt.twinx(ax=ax2)
    ax1.fill_between(x_sampled, ScaledLikelihood[:,1], alpha=0.4,label=r'$~Pr(X|\Theta=Positive_{geothermal}$)',color='green') #norm_pos1, InputMarg_weight
    ax1.plot(x_sampled, ScaledLikelihood[:,1],'g.') # norm_pos1
    ax1.fill_between(x_sampled,ScaledLikelihood[:,0], alpha=0.3,label=r'$~Pr(X|\Theta=Negative_{geothermal}$)',color='red') #norm_neg1 InputMarg_weight
    ax1.plot(x_sampled, ScaledLikelihood[:,0],'r.')  #norm_neg1
    ax1.legend(loc=0, fontsize=17)
    ax1.set_ylabel(' Scaled Likelihood $~Pr(x | y=Geothermal_{neg/pos}$', fontsize=25)#, rotation=-90)
    ax2.set_xlabel(str(x_cur), fontsize=18)
    ax1.tick_params(labelsize=20)
    ax_ylims = ax1.axes.get_ylim()  
    #print('ax_ylims',ax_ylims)
    # st.write('ax_ylims',ax_ylims)
    ax1.set_ylim(0,ax_ylims[1])
    # ax1.set_ylim(0,ax2_ylims[1])
    
    ax1.legend(fontsize=18)
    st.pyplot(fig20)

def Posterior_via_NaiveBayes(Pr_input_POS, X_train, X_test, y_train, y_test, x_sample, x_cur):
    """
    CURRENTLY not being used since it uses UN SCALED Likelihood 
    Function to calculate the posterior probability via Naive Bayes using prior from slide r

    Parameters
    PriorWeight: float, prior value from user input (order : NEG / POS) POSITIVE is second column! 
    X_train : array-like
        PROPORTION of x_sample??
    X_test : array-like, features in test
    y_train : array-like, labels in train
    y_test : array-like, labels in test
    x_sample : array-like
        full data attribute
    x_cur : parameter

    returns 
         post_input : array-like [len(x_sample) x 2]
            post_input[:,0] = probability of negative site using input prior
            post_input[:,1] = probability of positive site using input prior
         post_uniform : array-like [len(x_sample) x 2]
            post_uniform[:,0] = probability of negative site using 50/50 prior
            post_uniform[:,1] = probability of positive site using 50/50 prior
         
    """
    #   
    # # # # # # 
    model_NVML_input = GaussianNB(priors=[1-Pr_input_POS,Pr_input_POS,])
    #st.write('np.shape(X_train)',np.shape(X_train))
    model_NVML_input.fit(X_train.values[:,np.newaxis], y_train[:,np.newaxis]);

    model_NVML_uniform = GaussianNB(priors=[0.5,0.5])
    model_NVML_uniform.fit(X_train.values[:,np.newaxis], y_train[:,np.newaxis]);

    post_input = model_NVML_input.predict_log_proba(x_sample[:,np.newaxis])# X_test[:,np.newaxis])
    post_uniform = model_NVML_uniform.predict_log_proba(x_sample[:,np.newaxis])# X_test[:,np.newaxis])
    # st.write('post_input[:,0]',post_input[:,0])
    # st.write('post_input[:,1]',post_input[:,1])

    return post_input, post_uniform



def Posterior_Marginal_plot(post_input, post_uniform,marg,x_cur, x_sample):
    """
    Function plots the posterior values (y-axis) for (x_cur) data attribute values along x-axis at x_sample 
    post_input : array-like [len(x_sample) x 2]
        post_input[:,0] = probability of negative site using input prior
        post_input[:,1] = probability of positive site using input prior
    post_uniform : array-like [len(x_sample) x 2]
        post_uniform[:,0] = probability of negative site using 50/50 prior
        post_uniform[:,1] = probability of positive site using 50/50 prior
    marg : array-like [len(x_sample)]
        probability that each attribute value will occur given likelihood and prior scaling
    x_sample : array-like 
        Attribute values, sampled from minimum to maximum using ideal bandwidth (e.g. np.arange(min,max,ideal_bandwidth))
    """    
    
    fig4, axes = plt.subplots(figsize=(15,8),ncols=1,nrows=1)
    plt.plot(x_sample,post_input[:,1],color='purple', linewidth=6, alpha=0.7)
    plt.plot(x_sample,post_input[:,1],color='lime',linestyle='--', linewidth=3, label='$Pr(Positive|{})$ with Input Prior'.format(x_cur))
    plt.plot(x_sample,post_input[:,0],color='purple', linewidth=6)
    plt.plot(x_sample,post_input[:,0],'r--', linewidth=3,label='$Pr(Negative|{})$ with Input Prior'.format(x_cur))
    # plt.plot(x_sample,post_uniform[:,1],'g--', alpha=0.1, linewidth=3,label='$Pr(Postitive|{})$ with Uniform Prior'.format(x_cur))
    #plt.plot(x_sample,post_uniform[:,1],color='purple', alpha=0.1)
    plt.ylim([0,1])
    plt.legend(loc=2,fontsize=18,facecolor='w')#,draggable='True') 
    plt.xlabel(str(x_cur), fontsize=20)
    plt.ylabel('Posterior Probability', fontsize=20, color='purple')
    axes.tick_params(axis='x', which='both', labelsize=20)
    axes.tick_params(axis='y', which='both', labelsize=20, colors='purple')

    ax2 = axes.twinx()
    ax2.plot(x_sample,marg,color='orange',linestyle='dashdot', label='Marginal $Pr(X=x_j)$',alpha=0.7)
    ax2.fill_between(x_sample,marg, where=marg>=np.zeros(len(x_sample)), interpolate=True, color='orange',alpha=0.03)
    ax2.tick_params(axis='x', which='both', labelsize=20)
    ax2.tick_params(axis='y', which='both', colors='orange', labelsize=20)
    ax2.set_ylabel('Marginal Probability', color='orange',fontsize=20)
      
    # plt.legend(loc=1,fontsize=18) 
    st.pyplot(fig4)

    title = st.text_input('Filename', 'StreamlitImageDefault_{}.png'.format(x_cur))
    st.write('The current filename is', title)

    # SavePosteriorFig = st.checkbox('Please check if you want to save this figure')
    # if SavePosteriorFig:
    img = io.BytesIO()
    plt.savefig(img, format='png')
        
    btn = st.download_button(
        label="Download image "+title,
        data=img,
        file_name=title,
        mime="image/png"
        )


    return

def Posterior_by_hand(Pr_input_POS,Likelihood_pos, Likelihood_neg,x_sampled):
    """
    Calculate the Posterior from the Likelihood from KDE, no longer log-probability, properly normalized with 
    input posterior and resulting marginal.
    
    Parameters:
    Pr_input_POS : float, prior probability of positive geothermal
    Likelihood_pos : array-like [1 x 100]
    Likelihood_neg : array-like [1 x 100]
    x_sampled : array-like [1 x 100], sample of all possible data values
    """
    
    likelihood = np.transpose(np.vstack((Likelihood_neg, Likelihood_pos)))
    #st.write('np.sum(likelihood,1)',np.shape(likelihood),np.sum(likelihood,1))

    X_input_prior_weight_POS = np.outer(np.ones((np.shape(likelihood)[0],)),Pr_input_POS )
    X_input_prior_weight_NEG = np.outer(np.ones((np.shape(likelihood)[0],)),1.0-Pr_input_POS )
    X_input_prior_weight= np.hstack((X_input_prior_weight_NEG,X_input_prior_weight_POS))
    
    #st.write('Input Prior Weight array:', np.shape(X_input_prior_weight), X_input_prior_weight[0:10])
    Pr_InputMarg = np.sum(X_input_prior_weight * likelihood,1) # sum across model classes, columns

    X_unif_prior_weight = np.transpose(np.outer(np.ones((np.shape(likelihood)[1],)), 0.5))
    #print('Uniform Prior array:', X_unif_prior_weight)
    Pr_UnifMarg= np.sum(X_unif_prior_weight * likelihood,1)  # sum over model classes, columns
    
    #st.write('Pr_InputMarg',np.shape(Pr_InputMarg), np.sum(Pr_InputMarg))
    #st.write(Pr_InputMarg)
    # st.write('Pr_UnifMarg',Pr_UnifMarg)
  
    # POSTERIOR
    InputMarg_weight = np.kron(Pr_InputMarg[:,np.newaxis],np.ones((1,np.shape([1-Pr_input_POS,Pr_input_POS])[0]))) # should be num classes, num of Thetas
    UnifMarg_weight = np.kron(Pr_UnifMarg[:,np.newaxis],np.ones((1,np.shape([0.5,0.5])[0])))
    #st.write('marginals as 2d array InputMarg_weight',InputMarg_weight)
    InputMarg_weight[InputMarg_weight==0] = 1e-3  # Temporary fix to ensure no 0s
    
    Prm_d_Uniform = X_unif_prior_weight * likelihood / UnifMarg_weight
    Prm_d_Input = X_input_prior_weight * likelihood / InputMarg_weight
    #st.write('Prm_d_Input',Prm_d_Input)

    return Pr_InputMarg, Pr_UnifMarg, Prm_d_Input, Prm_d_Uniform

def marginal(Pr_prior_POS, predictedLikelihood_pos, predictedLikelihood_neg, x_sampled):
    """
     The marginal describes how frequent is each data bin. This function updates the marginal using the
     prior (input by user) and likelihood (from data selected)
      # # DO NOT USE mymodule.marginal( because it's passing unscaled likelihood!!!)
      
     Returns [1 X nbins] marginal
    """
    marg_input_POS = Pr_prior_POS * np.exp(predictedLikelihood_pos)
    marg_input_NEG = (1-Pr_prior_POS) * np.exp(predictedLikelihood_neg)
    marg_w = 1.0 / np.sum(marg_input_POS+marg_input_NEG)
    
    #    likesum = np.exp(predictedLikelihood_pos)+np.exp(predictedLikelihood_neg)
        # scale = 1.0/likesum
    figT, axes = plt.subplots(figsize=(15,8),ncols=1)
    axes.plot(x_sampled,np.exp(predictedLikelihood_pos),'.g')
    axes.plot(x_sampled,np.exp(predictedLikelihood_neg),'.r')
    axes.plot(x_sampled,marg_w*(marg_input_POS+marg_input_NEG),'*c')
    #st.pyplot(figT)
    #st.write('MARG SUM', np.sum(marg_w*(marg_input_POS+marg_input_NEG)))

    #return marg_w*np.vstack((marg_input_NEG, marg_input_POS)) ## what the hell is this?
    st.write('np.shape(Pr_d)',np.shape(marg_w*(marg_input_NEG+marg_input_POS)))
    return marg_w*(marg_input_NEG+marg_input_POS)