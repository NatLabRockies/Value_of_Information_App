from geophires_x_client import GeophiresXClient
from geophires_x_client.geophires_input_parameters import GeophiresInputParameters
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
import re

def Geophires_output(gradient,depth,type_geo,no_prod,no_inj):
    client = GeophiresXClient()
    result = client.get_geophires_result(
                GeophiresInputParameters({
                    "Gradient 1": gradient,
                    "Reservoir Depth": depth,
                    "End-Use Option": type_geo,
                    # "Power Plant Type": "1", #subcritical ORC, had "3" originally
                    #"Economic Model": "3",
                    "Starting Electricity Sale Price": "0.15",
                    "Ending Electricity Sale Price": "1.00",

                    #"Reservoir Heat Capacity": "790",
                    #"Reservoir Thermal Conductivity": "3.05",
                    #"Reservoir Porosity": "0.0118",
                    #"Reservoir Impedance": "0.01",
                    #"Number of Fractures": "108",
                    #"Fracture Shape": "4",
                    #"Fracture Height": "300",
                    #"Fracture Width": "400",
                    #"Fracture Separation": "30",
                    "Number of Production Wells": no_prod,
                    "Number of Injection Wells": no_inj, #Keep out for now, add in later
                })
            )



    with open(result.output_file_path, 'r') as f:
    #print(f.read())
    #print(f.read(1500))
        words = ['Project NPV:','Drilling and completion costs per well:']
                    
        lines = f.readlines()
        Pnum = [ind for ind, line in enumerate(lines) if "project npv" in line.casefold()][0]                 
        #num = 31#30            
                
        npv = str(lines[Pnum:Pnum+1]) # Drilling and completion costs
        npv1= npv.split(':')
        npv1 = npv.replace(" ","")
        npv1 = npv1.replace('\n',"")
        npv1 = npv1.split('MUSD')
        
        npv2 = npv1[0:1]
        
        npv2 = str(npv2)
        npvv = npv2.split(':')
        final_npv = npvv[1:2]
        aa = str(final_npv[0:1])
        val = (''.join(c for c in aa if (c.isdigit() or c =='.' or c =='-')))
        val2 = (val.strip())
    
        val2 = float(val2)
        print('NPV line 68, Pnum',val2, Pnum)
        # st.write('NPV line 85, num, Pnum',val2, num, Pnum)
        
        npv_final = val2*1e6
                       
        ## Drilling and completion costs per well       
        WellCostElecnum = [ind for ind, line in enumerate(lines) if "drilling and completion costs" in line.casefold()][0]
        # num = 96 # was 96, Change to 95 in new one
       
        dcpw = str(lines[WellCostElecnum:WellCostElecnum+1]) #num-1:num]) 
        print('npv line 112', dcpw)
        # stim1= stim.split(':')
        stim1 = dcpw.replace(" ","")
        stim1 = stim1.replace('\n',"")
        stim1 = stim1.split('MUSD')
        
        stim2 = stim1[0:1]
        
        stim2 = str(stim2)
        stim3 = stim2.split(':')
        final_stim = stim3[1:2]
        aa = str(final_stim[0:1])
        val = (''.join(c for c in aa if (c.isdigit() or c =='.' or c =='-')))
        # print('124 val', val)
        val2 = val.strip()

        val3 = re.sub(r"[^0-9\.]", "", val2)
        #print('type(val2) val3', val2, type(val2), val3)
        if len(val3)>0:
            val2 = float(val3)
            drill_cost = -1*val2*1e6

        # num = 94  # Drilling and completion costs for direct use (CHANGE VARIALBE NAME)
        # ddwell = str(lines[num-1:num])
        # print('ddwell line 120', ddwell)
    
    #     npv1= npv.split(':')
    #     npv1 = npv.replace(" ","")
    #     npv1 = npv1.replace('\n',"")
    #     npv1 = npv1.split('MUSD')
        
    #     npv2 = npv1[0:1]
        
    #     npv2 = str(npv2)
    #     npvv = npv2.split(':')
    #     final_npv = npvv[1:2]
    #     aa = str(final_npv[0:1])
    #     val = (''.join(c for c in aa if (c.isdigit() or c =='.' or c =='-')))
    #     val2 = (val.strip())
    #     val4 = re.sub("[^0-9\.]", "", val2)
    #     # print('149 npv2 val2 val4', npv2, val2, val4)
    #     if len(val4)>0:
    #         val2 = float(val4)
    #         drill_cost2 = -1*val2*1e6
    
    
    # if (type_geo == 2): #Direct use
    #     drill_cost = drill_cost2
    # else:
    #     drill_cost = drill_cost
        
    return npv_final,drill_cost