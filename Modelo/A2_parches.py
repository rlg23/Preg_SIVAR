import tensorflow as tf
import rioxarray as rxr
from tqdm import tqdm 
import xarray as xr 
import numpy as np 
import glob
import os 
import re


from functions import Data_nc 
from functions import guardado_parches as saving

g05=Data_nc( fname="GFS0.5_t2m_heightAboveGround_instant_2022_*.nc", path="../Datos/GFS/2022_std")
g05.path_collection()
g05.loadVar()

era5=Data_nc( fname="era5_2022*.nc", path="../Datos/ERA5/2022_std")
era5.path_collection()
era5.loadVar()

g05.upscaleVar(era5.values_t2m)

print("era5: ", era5.values_t2m.shape)
print("gfs: ", g05.values_t2m.shape)

#Para el guardado
primer_dia=re.findall(r'\d+', g05.path_ncfiles[171:354][0])[-1] 
ultimo_dia=re.findall(r'\d+', g05.path_ncfiles[171:354][-1])[-1] 
intervalo= primer_dia + "_" + ultimo_dia

print("Creacion de parches")

era5_parches = era5.extraccion_parches(Var=era5.values_t2m) #target
gfs_parches = g05.extraccion_parches(Var=g05.values_t2m)
lat_parches = era5.sampleLat()
lon_parches = era5.sampleLon()
orog_parches = era5.sampleBand1()
time1_parches,time2_parches = era5.sampleDayOfYear()
print("inp: ",gfs_parches.shape)
print("target: ",era5_parches.shape)
print("lat: ",lat_parches.shape)
print("long: ",lon_parches.shape)
print("orog: ",orog_parches.shape)
print("time1: ",time1_parches.shape)
print("time2: ",time2_parches.shape)

print("Chequeo de nulos")
g05.cantidad_nan(inputs=[era5_parches,gfs_parches,lat_parches,lon_parches,orog_parches,time1_parches,time2_parches])
print("Parches creados exitosamente")


#parche completo
saving(era5_parches,'era5_'+ intervalo,'parches',verbose=True)
saving(gfs_parches,'gfs_'+ intervalo,'parches',verbose=True)
saving(lat_parches,'lat_'+ intervalo,'parches',verbose=True)
saving(lon_parches,'lon_'+ intervalo,'parches',verbose=True)
saving(orog_parches,'orog_'+ intervalo,'parches',verbose=True)
saving(time1_parches,'time1_'+ intervalo,'parches',verbose=True)
saving(time2_parches,'time2_'+ intervalo,'parches',verbose=True)

print("Proceso de creacion de parches terminado")
