
import tensorflow as tf
import rioxarray as rxr
from tqdm import tqdm 
import xarray as xr 
import numpy as np 
import glob
import os 
import re
import matplotlib.pyplot as plt

class Data_nc:
    

    def __init__(self, fname="*", path='./', Var='t2m'):
        self.fname=fname
        self.path=path
        self.Var=Var
        self.path_ncfiles = []
        self.values=[]
        self.values_t2m=[]

    def path_collection(self):

        if '*' in self.fname:
        
            for year in os.listdir(self.path):
                single_path = os.path.join(self.path, year)
                self.path_ncfiles.append(single_path)

        else:
            
            self.path_ncfiles=os.path.join(self.path, self.fname)


        #for month in os.listdir(path_years):
            #path_months = os.path.join(path_years, month)
            #ncfiles = glob.glob(os.path.join(path_months, self.fname))
            #self.path_ncfiles.extend(ncfiles)
        return self.path_ncfiles

    def loadVar(self):
        '''Método que nos permite importar los distintos tipos de archivos (era5, dem, etc), estandarizar sus atributos internos y 
            retornar el DataArray definido por cfVarName (atributo self.values).'''
        
        if 'GFS' in self.path:

            data=[]
            for archivo in self.path_ncfiles[171:354]:
                ds=xr.open_mfdataset(archivo)
                ds_subset= ds.isel(valid_time=slice(0,8))
                data.append(ds_subset)

            self.values_t2m=xr.concat(data, dim="valid_time")['t2m']

        elif 'ERA5' in self.path:

            self.values = xr.open_mfdataset(self.path_ncfiles[171:354], concat_dim='time', combine='nested')
            self.values = self.values.rename( {'time':'valid_time'})
            self.values_t2m= self.values[self.Var][::3,:,:]

        elif 'DEM' in self.path or self.fname:

            self.values=xr.open_mfdataset(self.path_ncfiles, concat_dim='time', combine='nested')
            self.values_t2m= self.values[self.Var]
        
        return self.values_t2m.copy()

    def upscaleVar(self, other, method='nearest'):
        '''Método que interpola linealmente (default) las dimensiones del DataArray definido (values) de tal manera que tenga las mismas dimensiones
            que el DataArray dado
        inputs:
            other: [DataArray] DataArray sobre el cual se quiere compatibilizar las dimensiones
            method: [str] Método de interpolación, default='linear'
        return:
            self.values.copy(): [DataArray] Copia del DataArray sobre el cual se aplicó la interpolación y cuyas dimensiones son las mismas que 'other'.
        '''
        self.method = method
        self.values_t2m = self.values_t2m.interp_like(other, method=self.method)
        return self.values_t2m.copy() 

    def checking(self):
        time_dim= int( self.values_t2m.valid_time.shape[0] )
        return f'Cantidad de días importados: {time_dim // 8}'
    
    def extraccion_parches(self, Var, dx=32):

        '''Extrae parches de tamaño (3,32,32) y retorna el stack con todos los parches
    input:
        Var: [DataArray] Datarray retornado por getVar() de tamaño (40,401,101)
    return:
        stack: [tf tensor] Stack con los parches, stack.shape=(1368, 3, 32, 32)
        '''
    
        stack = tf.stack([],axis=0)
        dim_t= Var.shape[0]
        dim_lat= Var.shape[1]
        dim_lon= Var.shape[2]

        #recorremos en el tiempo
        for dt in tqdm(range(0,dim_t -2)):
            #recorremos en la latitude
            for dlat in range(0,dim_lat,dx):
                #recorremos en la longitude
                for dlong in range(0,dim_lon,dx):  
                    #Cortamos parches de 32x32 en 3 tiempos             
                    parche=Var[ dt:(dt+3), dlat:(dlat+dx), dlong:(dlong+dx)]

                    #si el stack no está vacío, concatenamos los parches
                    if tf.equal(tf.size(stack), 0) == False:
                        parche = tf.expand_dims(parche, axis=0)
                        stack= tf.concat((stack, parche), axis=0)
                        #y pasamos a la siguiente iteración
                        continue
                
                    #si el stack está vacío, lo inicializamos
                    stack=tf.stack([parche],axis=0)
        patches = tf.expand_dims(stack, 4) 
        print("Stack patches shape: ",patches.shape)   

        return patches
    
        ### Bloque 2.2: Latitude

    def sampleLat(self):
        ''' Compatibiliza y adapta las dimensiones de los valores de la latitud del DataArray (guardado en values) a las dimensiones
            del propio DataArray.
        return:
            self.extraccion_parches(Var=lat): [tensor] Retorna la salida de la función extracion_parches( Var=lan)
        '''
        ll = tf.constant(self.values_t2m.latitude.values)
        vv = self.values_t2m.shape
        lat = tf.transpose(tf.broadcast_to(ll, [vv[0], vv[2], vv[1]]), perm=(0, 2, 1))
        return self.extraccion_parches(Var=lat)
    ## --- ##

    #### Bloque 2.3: Longitude
    def sampleLon(self):
        ''' Compatibiliza y adapta las dimensiones de los valores de la longitud del DataArray (guardado en values) a las dimensiones
            del propio DataArray.
        return:
            self.extraccion_parches(Var=lon): [tensor] Retorna la salida de la función extracion_parches( Var=lon)
        '''
        ll = tf.constant(self.values_t2m.longitude.values)
        vv = self.values_t2m.shape         
        lon = tf.broadcast_to(ll, vv)
        return self.extraccion_parches(Var=lon)
    
    def sampleBand1(self):
        ''' Preprocesa los archivos relacionados a la orografía instanciándolos en una nueva clase y aplicando los mismos métodos que se emplearon para el input y el target 
            para posteriormente extraer sus parches.
            Parte del preprocesamiento conlleva reemplazar los datos Nan por ceros y concatenar los valores del tensor completo una cantidad de veces suficiente para que sus 
            dimensiones coincidan con las del DataArray.
        return:
            self.extraccion_parches(Var=stack_time): [tensor] Retorna la salida de la función extracion_parches( ) aplicada a los valores de stack_time.
        '''
        band1 = Data_nc("DEM_rep.nc", path='../Datos/dem', Var='Band1')
        band1.path_collection()
        band1.loadVar()
        band1.upscaleVar(self.values_t2m)
        stack_time=tf.repeat(band1.values_t2m, self.values_t2m.shape[0], axis=0)
        return band1.extraccion_parches(Var=stack_time)
    
    def sampleDayOfYear(self):
        ''' Método encargado de la adaptabilidad y compatibilidad de dimensiones de la variable 'valid_time' con el DataArray. Utiliza el tensor que contiene los nanosegundos transcurridos
            desde inicio de año hasta la fecha registrada en self.values.valid_time como base.
        return:
            time1 = self.extraccion_parches(Var=time) Retorna la salida de la función extracion_parches( ) aplicada a los valores de time.
            time2 = self.extraccion_parches(Var=time) Retorna la salida de la función extracion_parches( ) aplicada a los valores de time pero con un stride de 4.
        '''
        d = self.values_t2m.valid_time
        dd = (d-d.astype('M8[Y]')).astype(float) 
        ll = tf.constant(dd) 
        vv = self.values_t2m.shape 
        time = tf.transpose(tf.broadcast_to(ll, [vv[1], vv[2], vv[0]]), perm=(2,0,1))
        time1 = self.extraccion_parches(Var=time)
        time2 = self.extraccion_parches(Var=time, dx=4)

        return time1, time2 
        ## --- ##

    def cantidad_nan(self,inputs=[]):
        '''Recorre cada variable que conforma el input de la red, y cuenta los valores nan de cada uno.
        inputs:
            inputs: [List] Lista con los inputs de la red.
        '''
        inputs_name=["target","inp","lat","lot","orog","time1","time2"]
        cantidad=[];cont=0
        for inp in inputs:
            check_nan= tf.math.is_nan(inp[:,0,:,:,0])
            cont=np.sum(check_nan)
            cantidad.append(cont)

        for nombre,num in zip(inputs_name,cantidad):
            print(f"{nombre}- num valores nan: {num}")

    def normalization(self, data_in=input,inverse=False, scale_factor=[1,1]):
        '''Método que normaliza los valores entregados.
        input:
            data_in: [tensor] Tensor con los parches de la variable que forma parte del input de la Red
            inverse: [Boolean, default=False] True en caso de devolver la transformación
        return: 
            variable normalizada
        '''
        if not inverse:
            scale_factor_ = np.max(np.abs(data_in))
            scale_factor_2 = np.min(np.abs(data_in))
            data_out = (data_in-scale_factor_2)/(scale_factor_ - scale_factor_2)
            scale_factor_ = [scale_factor_,scale_factor_2]
        else:
            data_out=(data_in * (scale_factor[0]-scale_factor[1]))+scale_factor[1]
            scale_factor_ = scale_factor
        return data_out, scale_factor_
    
    def mysplit(self, inp, time, orog, lat, lon, target):
        ''' Método encargado de la división  de todas las variables que conforman el input de la red en: entrenamiento (60%), validación (20%) y testeo.
        return:
            división de cada variable en testeo, validación y testeo.
        '''
        data = tf.concat([tf.cast(inp, dtype=tf.float32),
                      tf.cast(time, dtype=tf.float32),
                      tf.cast(orog, dtype=tf.float32),
                      tf.cast(lat, dtype=tf.float32),
                      tf.cast(lon, dtype=tf.float32),
                      tf.cast(target, dtype=tf.float32)], axis=4)
        data = tf.random.shuffle(data, seed=10)
        n = data.shape[0]
        n_train = int(n * .6)
        n_valid = int(n * .2)
        n_test = n - n_train - n_valid
        data = tf.split(data, [n_train, n_valid, n_test])
        return (tf.expand_dims(data[0][:, :, :, :, 0],4),
            tf.expand_dims(data[0][:, :, :, :, 1],4),
            tf.expand_dims(data[0][:, :, :, :, 2],4),
            tf.expand_dims(data[0][:, :, :, :, 3],4),
            tf.expand_dims(data[0][:, :, :, :, 4],4),
            tf.expand_dims(tf.expand_dims(data[0][:,1,:,:,5],1),4),
            tf.expand_dims(data[1][:, :, :, :, 0],4),
            tf.expand_dims(data[1][:, :, :, :, 1],4),
            tf.expand_dims(data[1][:, :, :, :, 2],4),
            tf.expand_dims(data[1][:, :, :, :, 3],4),
            tf.expand_dims(data[1][:, :, :, :, 4],4),
            tf.expand_dims(tf.expand_dims(data[1][:,1,:,:,5],1),4),
            tf.expand_dims(data[2][:, :, :, :, 0],4),
            tf.expand_dims(data[2][:, :, :, :, 1],4),
            tf.expand_dims(data[2][:, :, :, :, 2],4),
            tf.expand_dims(data[2][:, :, :, :, 3],4),
            tf.expand_dims(data[2][:, :, :, :, 4],4),
            tf.expand_dims(tf.expand_dims(data[2][:,1,:,:,5],1),4))
    

def guardado_parches(parches, nombre_parches, carpeta, opcional='', verbose=False):
    ruta_completa = os.path.join(carpeta, nombre_parches + opcional + '.npy')
    np.save(ruta_completa, parches)
    if verbose:
        print("Parches guardados exitosamente en: ", ruta_completa)

def combined_image(tensor_pic):
    positions = [(0, 0), (0, 32), 
             (32, 0), (32, 32), 
             (64, 0), (64, 32), 
             (96, 0), (96, 32), 
             (128, 0), (128, 32), 
             (160, 0), (160, 32), 
             (192, 0), (192, 32)]

    image_size = 32
    combined_image = np.zeros((6 * image_size, 2 * image_size), dtype=np.float32)

    for i,j in enumerate( range(12)):
        tensor_data = tensor_pic[j].numpy()  
        combined_image[ positions[i][0]:positions[i][0] + image_size,  positions[i][1]:positions[i][1] + image_size ] = tensor_data   

    return combined_image  

def output_plot(input_plot,observed,fore):
    residuals = observed - fore
    
    params = {'axes.titlesize':'9',
              'xtick.labelsize':'9',
              'ytick.labelsize':'9',
              'figure.figsize': (20,20),
              'figure.dpi': 200
              }
    #plt.figure(figsize=[16,16])
    plt.style.use('ggplot')
    plt.rcParams.update(params)

    i=4
    plt.subplot(141) #131
    plt.imshow(fore, vmin = np.min([fore,observed]), vmax = np.max([fore,observed]))
    plt.colorbar(orientation="vertical",fraction=0.047, pad=0.01)
    plt.title('forecasted')

    plt.subplot(142) #132
    plt.imshow(residuals, vmin = np.min(residuals), vmax = np.max(residuals))
    plt.colorbar(orientation="vertical", fraction=0.047, pad=0.01)
    plt.title('residuals')


    plt.subplot(143) #132
    plt.imshow(observed, vmin = np.min([fore,observed]), vmax = np.max([fore,observed]))
    plt.colorbar(orientation="vertical",fraction=0.047, pad=0.01)
    plt.title('testing target')

    plt.subplot(144)
    plt.imshow(input_plot,
                    vmin = np.min([input_plot]),
                    vmax = np.max([input_plot]))
    plt.colorbar(orientation="vertical",fraction=0.047, pad=0.01)
    plt.title('input te')

    #plt.savefig('grad_loss.png')
    plt.show()
