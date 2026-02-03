import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import colors 
from matplotlib import ticker
import os
from PIL import Image
import requests
from io import BytesIO
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from geophires_x_client import GeophiresXClient
from geophires_x_client.geophires_input_parameters import GeophiresInputParameters

from GEOPHIRES_X import Geophires_output
from User_input import st_file_selector, Prior_probability_binary, make_value_array
from Naive_Bayes import make_train_test, optimal_bin
from Bayesian_Modeling import likelihood_KDE, Scaledlikelihood_KDE, Posterior_by_hand, Posterior_Marginal_plot, Posterior_via_NaiveBayes
from VOI import Vperfect, f_MI, f_VIMPERFECT, f_VPRIOR

# PRIORS - > USER INPUT
# st.header('Interactive Demonstration of Relationship between Value of Information and Prior Value')
st.header('Demo: Average outcome ($) making decision without additional data')



#Code below plots the Decision Tree image from github
# url = 'https://github.com/wtrainor/INGENIOUS_streamlit/raw/main/File Template/dtree2.png'
url = 'https://github.com/NREL/Value_of_Information_App/raw/main/File Template/dtree2.png'
# url = 'https://github.com/wtrainor/GR_Python_Workshop/raw/main/dtree.png'

#response = requests.get(url)
#image= Image.open(BytesIO(response.content))
#st.image(image, caption='Sample BinaryDecision Tree with Binary Geothermal Resource')

vprior_depth = np.array([1000,2000,3000,4000,5000,6000])

#st.write('What\'s the Prior Probability of a POSITIVE geothermal site?  $Pr(x=Positive)$')
#Pr_prior_POS_demo = mymodule.Prior_probability_binary() 


#### Value versus depth plot
count_ij = np.zeros((2,6))
value_array, value_array_df = make_value_array(count_ij, profit_drill_pos= 15e6, cost_drill_neg = -1e6)
# # st.write('value_array', value_array)
#Assigning values that match GEOPHIRES drilling costs.
value_drill_DRYHOLE = np.array([-1.9e6, -2.8e6, -4.11e6, -5.81e6, -7.9e6, -10.4e6])

vprior_depth = np.array([1000,2000,3000,4000,5000,6000])

firstfig, ax = plt.subplots()
#firstfig1, axe = plt.subplots(1,2)
plt.plot(vprior_depth,value_drill_DRYHOLE,'g.-', linewidth=5,label='$V_{prior}$')
plt.ylabel(r'Average Drilling Cost [\$]',fontsize=14)
plt.xlabel('Depth (m)', color='darkred',fontsize=14)
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)
# ax.yaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter('${x:0,.0f}') #:0,.0f
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_formatter('{x:0,.0f}')


# Code for table with decision economic outcomes defined by the user.
newValuedf1 = pd.DataFrame({
               "action": [ 'walk away','drill',],               
                "Positive: Geothermal Resource Exists": [0,value_array_df.iloc[1,1]*10],
                "Negative: Absence of Geothermal": [0,'User Defined Revenue - drilling costs']}   
        )

# list = 
# idx= pd.Index(list)
# newValuedf.set_index(idx)
newValuedf1.style.set_properties(**{'font-size': '35pt'}) # this doesn't seem to work
 #bigdf.style.background_gradient(cmap, axis=1)\

# Code to input these values
original_title = '<p style="font-family:Courier; color:Black; font-size: 25px;"> Enter revenue for your \'drill\' decision with *positive* geothermal combination [$] </p>'
st.markdown(original_title, unsafe_allow_html=True)
edited_df = st.data_editor(newValuedf1,hide_index=True,width='stretch')
# try this https://discuss.streamlit.io/t/center-dataframe-header/51193/4
# newValuedf1.style.set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
# st.markdown('<style>.col_heading{text-align: center;}</style>', unsafe_allow_html=True)
# newValuedf1.columns = ['<div class="col_heading">'+col+'</div>' for col in newValuedf1.columns] 
# st.write(newValuedf1.to_html(escape=False), unsafe_allow_html=True)

revenue = np.ravel(edited_df[['Positive: Geothermal Resource Exists']])
# st.write('revenue', revenue)
pos_outcome = float(revenue[1])  # float(revenue.values[1])
# pos_outcome = float(edited_df[['Positive: Geothermal Resource Exists']].values[1])
# st.write('pos_outcome',pos_outcome)
#neg = float(edited_df[['No Hydrothermal Resource (negative)']].values[1])
value_array, value_array_df = make_value_array(count_ij, profit_drill_pos= pos_outcome, cost_drill_neg = -1e-6)


## Calculate Vprior
Pr_prior_POS_demo = Prior_probability_binary() 
## Find Min Max for the Vprior Demo plot
vprior_INPUT_min = f_VPRIOR([0.9,0.1], value_array, value_drill_DRYHOLE[-1])  
vprior_INPUT_max = f_VPRIOR([0.9,0.1], value_array, value_drill_DRYHOLE[0])   
VPI_max = Vperfect(Pr_prior_POS_demo, value_array,  value_drill_DRYHOLE[0])  

vprior_INPUT_demo_list = list(map(lambda vv: f_VPRIOR([1-Pr_prior_POS_demo,Pr_prior_POS_demo], 
                                                              value_array,vv),value_drill_DRYHOLE))
# first_sub = '<p style="font-family:Courier; color:Black; font-size: 30px;"> $Pr(Success) = Pr(Geothermal=Positive)=$</p>'
# +str(Pr_prior_POS_demo) 
# st.markdown(first_sub, unsafe_allow_html=True)
st.latex(r'''Pr(Success) = Pr(Geothermal=Positive)='''+str(Pr_prior_POS_demo), help='Set with slider above')  
st.write('$V_{prior} =$  best action given each weighted average')
# st.markdown("""
# <style>
# .big-latex {
#     font-size:80px !important;
#     font-family:Courier;
# }
# </style>""", unsafe_allow_html=True)

##st.write('Average outcome, using $Pr(Success)$ ~ Prior probability')
## st.write(r'''$V_{prior} =  \max\limits_a \Sigma_{i=1}^2 Pr(\Theta = \theta_i)  v_a(\theta_i) \ \  \forall a $''')

# stuff = '''$V_{prior}=$'''
# st.markdown(stuff, unsafe_allow_html=True) #'<p class="big-latex"> stuff </p>'
# st.latex(r''' \max\limits_a 
#             \begin{cases}
            
#             Pr(positive) v_{drill}(positive) + Pr(negative)  v_{drill}(negative)             &\text{if a=drill} \\
#             &\ \\
#             Pr(positive) v_{nothing}(positive) + Pr(negative)  v_{nothing}(negative)  =0     &\text{if do a=nothing}
#             \end{cases} ''')

         
# Plotting VOI
showVperfect = st.checkbox('Show Vperfect')

# Plotting Depth vs Value of Information
#showVperfect2 = st.checkbox('Show Vperfect')
firstfig2, ax1 = plt.subplots() # Plotting the VOI figure

ax1.plot(vprior_depth, vprior_INPUT_demo_list, 'g.-', linewidth=5,label='$V_{prior}$')
plt.ylabel(r'Average Outcome Value [\$]',fontsize=14)
plt.xlabel('Well Depth (m)', color='brown',fontsize=14)

# Plotting text on the VOI plot
txtonplot = r'$v_{a=Drill}(\Theta=Positive) =$'
ax1.text(np.min(vprior_depth), value_array[-1,-1]*0.7, txtonplot+r'\${:0,.0f}'.format(value_array[-1,-1]), 
        size=12, color='green',
         #va="baseline", ha="left", multialignment="left",
          horizontalalignment='left',
         verticalalignment='top')#, bbox=dict(fc="none"))

# Plotting the inset axes with drilling cost curve

if showVperfect:  
    VPIlist = list(map(lambda uu: Vperfect(Pr_prior_POS_demo, value_array,uu),value_drill_DRYHOLE))
    # st.write('VPI',np.array(VPIlist),vprior_INPUT_demo_list)
    VOIperfect = np.maximum((np.array(VPIlist)-np.array(vprior_INPUT_demo_list)),np.zeros(len(vprior_INPUT_demo_list)))
    # VPI_list = list(map(lambda v: mymodule.f_Vperfect(Pr_prior_POS_demo, value_array, v), value_drill_DRYHOLE))
    ax1.plot(vprior_depth,VPIlist,'b', linewidth=5, alpha=0.5, label='$V_{perfect}$')
    ax1.plot(vprior_depth,VOIperfect,'b--', label='$VOI_{perfect}$')

plt.legend(loc=1)
plt.ylim([vprior_INPUT_min,value_array[-1,-1]*0.8]) # YLIM was (VPI_max+20)
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)
# ax.yaxis.set_major_formatter(formatter)
ax1.yaxis.set_major_formatter('${x:0,.0f}') #:0,.0f
ax1.xaxis.set_major_formatter(formatter)
ax1.xaxis.set_major_formatter('{x:0,.0f}')

axins1 = inset_axes(
    ax1,
    width="28%",  # width: 50% of parent_bbox width
    height="28%",  # height: 5%
    loc="center right",
)

axins1.plot(vprior_depth,value_drill_DRYHOLE,'g.-', linewidth=5)#,color = 'red')

#plt.ylabel(r'Average Drilling Cost [\$]',fontsize=7)
plt.xlabel('Depth (m)', color='brown',fontsize=7)
plt.title(r'Drilling Costs [\$]', fontsize = 7)
formatter = ticker.ScalarFormatter()
formatter.set_scientific(True)
axins1.yaxis.set_major_formatter(formatter)
axins1.yaxis.set_major_formatter('${x:0,.0f}') #:0,.0e
axins1.tick_params(axis='y', colors='red')
axins1.xaxis.set_major_formatter(formatter)
axins1.xaxis.set_major_formatter('{x:0,.0f}')

#Show the VOI plot
st.pyplot(firstfig2)

if showVperfect:  
    
    st.write(r'When you "know" when either subsurface condition occurs, you can pick the best ($\max\limits_a$) drilling altervative first ($v_a$).')
    st.write(r'''$V_{perfect} =  \Sigma_{i=1}^2 Pr(\Theta = \theta_i) \max\limits_a v_a(\theta_i) \ \  \forall a $''')
    st.write(r'''$VOI_{perfect} (Value \ of \ Information) = V_{perfect}-V_{prior}=$'''+str(VPIlist[0])+' - '+str(vprior_INPUT_demo_list[0]))

st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True) # Code to draw line to separate demo problem from main part


# Sidebar is where user picks data file
with st.sidebar:
    attribute0 = None        
    # LOCATION OF THIS FILE 
    st.page_link("https://forms.office.com/Pages/ResponsePage.aspx?id=fp3yoM0oVE-EQniFrufAgDjT0ckom9BErLMgwLSsipBUNDkwR05DRjlFNVpMQlFTQkxKSVA3NEJYMi4u",\
                 label=':blue-background[**Click here:\n for feedback**]',icon=":material/cloud:")

    st.page_link("https://github.com/NREL/Value_of_Information_App/tree/main/File%20Template",\
                 label=':orange-background[**Click here:\n User Manual, file templates & examples**]',icon=":material/question_exchange:")
    uploaded_files = st.file_uploader(\
        "Upload two data files,namely a Positive Label file (\'POS_\' :fire:) & a Negative Label (\'NEG_\':thumbsdown:) file ", \
                                      type=['csv'],accept_multiple_files=True)
    
    count_neg= 0
    count_pos = 0
    if uploaded_files is not None and len(uploaded_files)==2:
        st.header('VOI APP')
        st.subheader('App Data')

        st.subheader('Choose Geothermal type')
        types = ['Electricity','Direct-Use']
        geo_choice = st.selectbox('Which end use option do you wish to explore?', types)
        

        st.subheader('Choose attribute for VOI calculation')
        
        for uploaded_file in uploaded_files:
            # bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            if (uploaded_file.name[0:3]=='POS') :
                if ( count_pos == 1):
                    st.write('You didn\'t select a NEG file, try again')
                else:
                    pos_upload_file = uploaded_file
                    df = pd.read_csv(pos_upload_file)
            #       st.write('attribute0 is None',attribute0==None, not attribute0)
            #       if not attribute0:
                    attribute0 = st.selectbox('Which attribute would you like to explore?', df.columns)
                   
                    count_pos = count_pos + 1
            
            elif (uploaded_file.name[0:3]=='NEG'):
                if ( count_neg == 1):
                    st.write('You didn\'t select a POS file, try again')
                else:
                    neg_upload_file = uploaded_file
                    dfN = pd.read_csv(neg_upload_file)
                    count_neg = count_neg + 1       
            else:
                if ( uploaded_file.name[0:3] == 'NEG'):
                    st.write('You didn\'t select a POS file, try again')
                else:
                    st.write('You didn\'t select a NEG file, try again')

                
        if pos_upload_file.name[3:7] != neg_upload_file.name[3:7]:
                st.write('You aren\'t comparing data from the same region. STOP!')
        else:        
            st.write('POS File summary...')
            st.write(df.describe())        
            st.write('NEG File preview...')
            st.write(dfN.describe())   
        
    #with st.spinner("Loading..."):
    #    time.sleep(5)
    #st.success("Done!")
    else:
        st.write('please upload file')


# Compute Value of IMPERFECT information (VOIimperfect) with uploaded data
if uploaded_files is not None:
    st.title('Main App: ')
             
    if attribute0 is not None:
        st.title('Likelihoods from uploaded data')
        
        x_cur = attribute0
    
        df_screen = df[df[x_cur]>-9999]
        df_screenN = dfN[dfN[x_cur]>-9999]
        #st.write('dataframe is shape: {thesize}'.format(thesize=df_screenN.shape))
        #st.write('attribute stats ', df_screen[attribute0].describe())

        neg_site_col_name = 'NegSite_Distance' # I change the name when making csv 'tsite_dist_nvml_neg_conus117_250m' #
        distance_meters = st.slider('Thresholding distances of data to labels (meters for INGENIOUS data sets)',
                                    10, int(np.max(df_screen['PosSite_Distance'])-10), int(np.max(df_screen['PosSite_Distance'].quantile(0.1))), step=100) # min, max, default
        # NEG_distance_meters = st.slider('Change likelihood by *screening* distance to negative label [km or meters??]', 
        #     10, int(np.max(df_screenN['NegSite_Di'])-10), int(np.median(df_screenN['NegSite_Di'])), step=1000)

        # round to make sure it rounds to nearest 10
        dfpair0 = df_screen[(df_screen['PosSite_Distance'] <=round(distance_meters,-1))] 
        
        dfpair = dfpair0[dfpair0[x_cur]>-9999] 
        # # # OJO : may want to keep this off until have it for NEG 
        dfpairN = df_screenN[(df_screenN[neg_site_col_name ] <=round(distance_meters,-1))] 
        
        if np.shape(dfpairN)[0]==0:
            st.write('using Q1 distance for Negative sites')
            dfpairN = df_screenN[(df_screenN[neg_site_col_name ] <= np.percentile(df_screenN[neg_site_col_name ],10))] 
        
        st.subheader('Empirical Likelihoods: bin counts of data')
       
        #waiting_condition = 1
        #while (waiting_condition):
        #    st.image('https://media.giphy.com/media/gu9XBXiz60HlO5p9Nz/giphy.gif')

        # waiting_condition = mymodule.my_kdeplot(dfpair,x_cur,y_cur0,y_cur1,waiting_condition)
        
        # split up if we want to test bandwidth 
        X_train, X_test, y_train, y_test = make_train_test(dfpair,x_cur,dfpairN)
 
        best_params, accuracy = optimal_bin(X_train, y_train)

        # Likelihood via KDE estimate
        predictedLikelihood_pos, predictedLikelihood_neg, x_sampled, count_ij= likelihood_KDE(X_train,X_test, y_train, y_test,x_cur, best_params)
     
        st.subheader(':blue[Prior]-Scaled Likelihood') 
        Pr_prior_POS = Prior_probability_binary('Prior used in Posterior')
        # st.write(':blue['+r'''$Pr(\Theta = \theta_i)$'''+'] in posterior')
                 
        # New plot for normalized likelihood: Modeled after Likelihood via KDE estimate
        Scaledlikelihood_KDE(Pr_prior_POS,predictedLikelihood_neg, predictedLikelihood_pos,X_train,X_test, y_train, y_test,x_cur,x_sampled, best_params)
            
        st.subheader(':point_down: :violet[Posterior] ~=:blue[Prior] x Likelhood ')
        st.latex(r'''\color{purple} Pr( \Theta = \theta_i | X =x_j ) = \color{blue}
        \frac{Pr(\Theta = \theta_i ) \color{black} Pr( X=x_j | \Theta = \theta_i )}{\color{orange} Pr (X=x_j)}''')  
        # POSTERIOR via_Naive_Bayes: Draw back here the marginal not using scaled likelihood..
        post_input, post_uniform = Posterior_via_NaiveBayes(Pr_prior_POS,X_train, X_test, y_train, y_test, x_sampled, x_cur)
             
       
        Pr_InputMarg, Pr_UnifMarg, Prm_d_Input, Prm_d_Uniform = Posterior_by_hand(Pr_prior_POS,predictedLikelihood_pos, predictedLikelihood_neg, x_sampled)
        Posterior_Marginal_plot(Prm_d_Input, Prm_d_Uniform, Pr_InputMarg, x_cur, x_sampled) # WAS inputting: post_input, post_uniform, Pr_Marg, x_cur, x_sampled)

        # # # # # # VALUE OUTCOMES # # # # # # # # # #
        st.header('How much is this imperfect data worth?')
        Input_title = '<p style="font-family:Courier; color:Green; font-size: 30px;"> Enter gradient and depth</p>'
        st.markdown(Input_title, unsafe_allow_html=True)
        inputs = pd.DataFrame({
               "action": ['Gradient [C/km]','Depth [km]'],               
                "Values": [30,3]}   
        )

        input_df = st.data_editor(inputs,hide_index=True,width='stretch')
        gradient = input_df['Values'].values[0]
        depth = input_df['Values'].values[1]

        # GEOPHIRES economics part
        type_geo = 1
        if (geo_choice=='Direct-Use'):
            type_geo = 2
        else:
            type_geo = 1
        
        # Pulling in NPV and drilling costs from GEOPHIRES-X
        no_prod = 2
        no_inj = 2 # Default production and injection wells from GEOPHIRES- user can change if needed.
        npv_final,drill_cost = Geophires_output(gradient,depth,type_geo,no_prod,no_inj)

        # Table for Outcomes part with GEOPHIRES costs
        newValuedf = pd.DataFrame({
               "action": ['walk away','drill'],
               "Geothermal Resource Exists (positive)": [0,npv_final],
               "No Geothermal Resource Exists (negative)": [0,drill_cost]}   
        )

        # list = 
        # idx= pd.Index(list)
        # newValuedf.set_index(idx)
        newValuedf.style.set_properties(**{'font-size': '35pt'}) # this doesn't seem to work
        #bigdf.style.background_gradient(cmap, axis=1)\

        # Code to be written to input these values
        original_title = '<p style="font-family:Courier; color:Green; font-size: 15px;"> Enter revenue for your drill decision with positive geothermal combination</p>'
        
        st.markdown(original_title, unsafe_allow_html=True)
        st.write("Default values are based on GEOPHIRES results for gradient and depth (user input)")
        edited_df = st.data_editor(newValuedf,hide_index=True,width='stretch')

        pos_drill_outcome = float(np.ravel(edited_df[['Geothermal Resource Exists (positive)']])[1]) 
        neg_drill_outcome = float(np.ravel(edited_df[['No Geothermal Resource Exists (negative)']])[1])
        # st.write('old and new', float(edited_df[['Geothermal Resource Exists (positive)']].values[1]), pos_drill_outcome)
        # st.write('old and new', float(edited_df[['No Geothermal Resource Exists (negative)']].values[1]), neg_drill_outcome)
        value_array, value_array_df = make_value_array(count_ij, profit_drill_pos= pos_drill_outcome, cost_drill_neg = neg_drill_outcome) # Karthik Changed here to reflect new values
        #st.write('value_array', value_array)

        #f_VPRIOR(X_unif_prior, value_array, value_drill_DRYHOLE[-1])  
        value_drill_DRYHOLE = np.linspace(100, -1e6,10)

        # This function can be called with multiple values of "dry hole"
        vprior_unif_out = f_VPRIOR([1-Pr_prior_POS,Pr_prior_POS], value_array) #, value_drill_DRYHOLE[-1]       
                       
        #st.subheader(r'''$V_{prior}$ '''+'${:0,.0f}'.format(vprior_unif_out).replace('$-','-$'))

        VPI = Vperfect(Pr_prior_POS, value_array)
        # st.subheader(r'''$VOI_{perfect}$ ='''+str(locale.currency(VPI, grouping=True )))
        #st.subheader('Vprior  \${:0,.0f},\t   VOIperfect = \${:0,.0f}'.format(vprior_unif_out,VPI).replace('$-','-$'))
        
        
        # VII_unif = mymodule.f_VIMPERFECT(post_uniform, value_array,Pr_UnifMarg)
        VII_input = f_VIMPERFECT(Prm_d_Input, value_array, Pr_InputMarg)
        VII_unifPrior = f_VIMPERFECT(Prm_d_Uniform, value_array, Pr_UnifMarg)
                       
        
        st.subheader(r'''$V_{imperfect}$='''+'${:0,.0f}'.format(VII_input).replace('$-','-$'))
        st.subheader(r'Vprior  \${:0,.0f},\t   VOIperfect = \${:0,.0f}'.format(vprior_unif_out,VPI-vprior_unif_out).replace('$-','-$'))
        st.subheader(r'''$V_{perfect}$='''+'${:0,.0f}'.format(VPI).replace('$-','-$'))
        # st.write('with uniform marginal', locale.currency(VII_unifMarginal, grouping=True ))
        # st.write('with uniform Prior', '${:0,.0f}'.format(VII_unifPrior).replace('$-','-$'))
        
        MI_post, NMI_post = f_MI(Prm_d_Input,Pr_InputMarg)
        #Basic question: How far apart (different) are two distributions P and Q? Measured through distance & divergences
        #https://nobel.web.unc.edu/wp-content/uploads/sites/13591/2020/11/Distance-Divergence.pdf
        # st.write('Mutual Information:', MI_post)
        # st.write('Normalized Mutual Information:', NMI_post)
        # st.write(accuracy,(VII_input,MI_post,accuracy)) #['bandwidth']
        dataframe4clipboard = pd.DataFrame([[VII_input,NMI_post,accuracy]])#,  columns=['VII','NMI','accuracy'])
        #st.write(dataframe4clipboard)
       #dataframe4clipboard.to_clipboard(excel=True,index=False)

    else: 
        st.write("Please upload data files on left")
else:
    st.write("Please upload any data.")
        