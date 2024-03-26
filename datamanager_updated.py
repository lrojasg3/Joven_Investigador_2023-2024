import pandas as pd
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, std, percentile
from scipy import stats
import seaborn as sns
import os
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pymongo import MongoClient

#-------------------------------------------------
# help
#-------------------------------------------------

"""
.. Label, use :ref:`text <label>` for reference
.. _datamanager:




Classes
=======


"""




class DataManager:
    """
    El modulo :py:mod:`datamanager` proporciona las herramientas para cargar los datos de las unidades simple y demas dispositivos. El modulo igualmente permite
    realizar preprocesamiento de los datos cargados.

    Clase base de datamanager. Esta clase se utiliza como entrada base para los demas modulos.

    """
    def __init__(self,path = '',dataframe="deafault"):
        """
        Crear datamanager. El datamanager puede ser creado a partir de un objeto dataframe (pandas), cargando los archivos desde un .csv o realizando una peticion
        de datos al servicio de SUMMAN.

        Arguments:

        * ``dataframe`` : Opcional. Objeto pandas dataframe con los datos de la unidad Simple.
        * ``path`` : Opcional. Objeto tipo str, donde estan los archivos ".csv" para ser cargadas como un dataframe

        """

        if isinstance(dataframe,pd.DataFrame):

            print('entre')
            self.data=dataframe
            
        elif isinstance(path, str):
            #if not(dataframe=='default'):
            #    print('Variable is not a Pandas DataFrame')
            self.path = path
            self.data = []
            list_file = [self.path + f for f in os.listdir(self.path) if f.endswith(('.CSV','.csv'))]
            for i in range(len(list_file)):
                print(list_file[i])
                df = pd.read_csv(f'{list_file[i]}')
                #Garantizamos eliminar las filas que tienen los nombres de columnas
                df = df[df['time'] != 'time']
                #Creamos la fecha a partir del nombre del archivo
                fecha = f'20{list_file[i][-6:-4]}-{list_file[i][-8:-6]}-{list_file[i][-10:-8]} '
                df['Fecha'] = fecha
                
                self.data.append(df) 
            
            self.data = pd.concat(self.data)
            self.data['Fecha_Hora'] = self.data.Fecha + self.data.time #Creamos un vector de fecha y hora    
            self.data.index = self.data['Fecha_Hora']
            self.data.index = pd.to_datetime(self.data.index)
            self.data.drop(['Fecha_Hora','Fecha'], axis = 1, inplace = True) #Eliminamos las columnas que no sirven

    def load_data(self):
        """
        Cargar datos desde un archivo csv. Se crea la variable ``data`` tipo dataframe de pandas con los datos cargados. El archivo csv debe tener como index una variable tipo datetime, o alguna columna con nombre timeStamp,time,dates, Fecha_hora.
        Si alguna de estas columnas existe, automaticamente se creara el index con formato datetime.
        En caso de no instanciar la clase con un objeto pandas, sino con una ruta, se creará el objetivo pandas, con los archivos ".csv" contenidos en la ruta que instanció la clase
        
        
        Argumentos:
        
        * ``file_name`` : Nombre del archivo csv con los datos a cargar.
        * ``deliminiter`` : Caracter deliminator de columnas en el archivo csv. Por defecto.

        * ``Nota`` : Sino se ingresan variables y se inicializa la clase con una ruta, se creará un DataFrame con los archivos ".csv" que contenga la ruta entregada.


        
        """

        try:
            #self.data = pd.read_csv(file_name,delimiter=delimiter) 
            
            if self.check_variable('timeStamp'):
                self.data['dates']=[datetime.fromtimestamp(int(fs/1000)) for fs in self.data['timeStamp']]
                self.data['dates']=self.data['dates'].round('min')
                self.data.sort_values(by='dates',ascending=True,inplace=True)
                self.data.reset_index(drop=True, inplace=True)
                self.data = set.index('dates',inplace=True)
                # self.data=self.data.resample('T', label='right').mean()
            if self.check_variable('time'):
                self.data['time']=[datetime.strptime(fs,'%Y-%m-%d %H:%M:%S') for fs in self.data['time']]
                self.data.sort_values(by='time',ascending=True,inplace=True)
                self.data.reset_index(drop=True, inplace=True)
                self.data.set_index('time',inplace=True)
                # self.data=self.data.resample('T', label='right').mean()
            if self.check_variable('dates'):
                self.data['dates']=pd.to_datetime(self.data['dates'])
                self.data.sort_values(by='dates',ascending=True,inplace=True)
                self.data.reset_index(drop=True, inplace=True)
                self.data.set_index('dates',inplace=True)
                # self.data=self.data.resample('T', label='right').mean()
            if self.check_variable('Fecha_hora'):
                self.rename_variable('Fecha_hora','Fecha_Hora')
            if self.check_variable('Fecha_Hora'):
                self.data['Fecha_Hora']=[datetime.strptime(fs,'%Y-%m-%d %H:%M:%S') for fs in self.data['Fecha_Hora']]
                self.data.sort_values(by='Fecha_Hora',ascending=True,inplace=True)
                self.data.reset_index(drop=True, inplace=True)
                self.data.set_index('Fecha_Hora',inplace=True)    
                #self.data=self.data.resample('T', label='right').mean()

        except: 
            
            for i in self.data.keys():
                try:
                    self.data[i] = self.data[i].astype(float)
                except:
                    pass

        return self.data
    
    
    def clean_variable(self,variable,max_value=20000000,min_value=-99999,replace_min=np.NaN, replace_max=np.NaN):
        """
        Reemplazar los valores menores a min_value y mayores a max_value por replace_min y replace_max respectivamente.
        Esta funcion se puede utilizar para por ejemplo, convertir en NaN los datos no validos de algunas fuentes de informacion 
        (normalmente -99999)
        
        
        Argumentos:
        
        * ``variable`` : Nombre de la variable de la unidad Simple a procesar.
        * ``max_value`` : Valor maximo admisible. Los valores mayores a max_value seran reemplazados por replace_max
        * ``min_value`` : Valor minimo admisible. Los valores menores a min_value seran reemplazados por replace_min
        * ``replace_max`` : Valor de reemplazo para aquellos valores mayores a max_value
        * ``replace_min`` : Valor de reemplazo para aquellos valores menores a min_value
        """
        if self.check_variable(variable):
            self.data[variable].values[self.data[variable].values>max_value]=replace_max
            self.data[variable].values[self.data[variable].values<min_value]=replace_min
            
        else:
            raise ValueError("Data Frame does not contain required variable")
    
    def remove_variable(self,variable):
        """
        Borrar la variable definida del datamanager.
        
        
        Argumentos:
        
        * ``variable`` : Nombre de la variable de la unidad Simple a borrar.
        """
        if self.check_variable(variable):
            self.data.drop(variable,inplace=True, axis=1)

        else:
            raise ValueError("Data Frame does not contain required variable")

    def calc_conf_interval(self, variable='', alpha=0.95):
        """
        Calcular el intervalo de confianza de la variable definida. El metodo empleado es el calculo de intervalos de confianza Bayesiano para la 
        media, varianza y desviacion estandar.
        
        Argumentos:
        
        * ``variable`` : Nombre de la variable de la unidad Simple a procesar.
        * ``alpha`` : Proabilidad de que el intervalo de confianza calculado contenga el valor verdadero. Por defecto 0.95.
        
        Retorna:
            
        * ``mean`` : Intervalo de confianza para la media.  
        * ``var`` : Intervalo de confianza para la varianza.  
        * ``std`` : Intervalo de confianza para la desviacion estandar.  
        
        """
        
        if self.check_variable(variable):
          
           data_variable = self.data[variable].values
           #removemos posibles nan
           data_variable = data_variable[~np.isnan(data_variable)]
           mean, var, std = stats.bayes_mvs(data_variable)

           return mean,var,std
        else:
            raise ValueError("Data Frame does not contain required variable")
    
    
    def rename_variable(self,variable,new_name):
        """
        Renombrar una variable dentro del datamanager.
        
        Argumentos:
        
        * ``variable`` : Nombre de la variable de la unidad Simple a procesar.
        * ``new_name`` : Nuevo nombre de la variable
        
        
        """
        if self.check_variable(variable):
            self.data.rename(columns={variable:new_name},inplace=True)
            
        else:
            raise ValueError("Data Frame does not contain required variable")
    
    def variables(self):
        """
        Mostrar las variables disponibles en el datamanager.
        
        """
        df = self.load_data()
        return list(df.columns) 
    
    
    def check_variable(self,variable):
        """
        Comprobar si una variable existe en el datamanager.
        
        Argumentos:
        
        * ``variable`` : Nombre de la variable de la unidad Simple a procesar.
        
        Retorna:
            
        * ``check`` :  True si la variable existe, False si la variable no existe. 
        
        
        """
        
        if variable in self.data.columns:
            check=True
        else:
            check=False
            
        return check
    
    def see_variable(self,variable,ini=0,end=10):
        """
        Mostrar los valores de la variable entre ini y end.
        Argumentos:
        
        * ``variable`` : Nombre de la variable de la unidad Simple a procesar.
        * ``ini`` : Posicion inicial a mostrar. Por defecto 0.
        * ``end`` : Posicion final a mostrar. Por defecto 10.
        
        """
        if self.check_variable(variable):
            aux=self.get_values(variable)
            print(aux[ini:end])
            return aux[ini:end]
        else:
            raise ValueError("Data Frame does not contain required variable")
    
    def get_values(self,variable):
        """
        Obtener los valores de una variable para por ejemplo, crear una variable externa al datamanager.
        
        Argumentos:
        
        * ``variable`` : Nombre de la variable de la unidad Simple a procesar.
        
       
        
        """
        if self.check_variable(variable):
            return self.data[variable].values
        else:
            raise ValueError("Data Frame does not contain required variable")   

    def graphic(self, data, variable = 'CO [ppm]', unit = 'ppm',
                shape_figs = (1,1),  position = [0,0] ,color = '#056674', fig = plt.figure(figsize=(10,5))):
    
        """
        Generar 1 o más gráficas en un sólo mosáico.
        Argumentos:
        
        * ``data`` : DataFrame con los datos.
        * ``variable`` : Variable a graficar.
        * ``unit`` : Unidades de la variable
        * ``labelsize`` :  Tamaño de fuentes   
        * ``shape_figs`` : shape de la matriz de figura en forma i,j. Por defecto (1,1)
        * ``position`` : Posición de la figura en la matriz en forma i,j. Por defecto [0,0]
        * ``color`` : Color de la curva de la serie de tiempo. Por defecto #056674
        * ``fig`` : objeto de matplotlib
        * ``loc`` : ubicación de la leyenda. Por defecto 0 (best).
        
        Retorna:
                
            * ``fig`` :  Figura.
        """
        #print(data)
        # fig = fig
        gs = gridspec.GridSpec(shape_figs[0],shape_figs[1])

        #data.index = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in data.index]

        ax1=fig.add_subplot(gs[position[0],position[1]])

        minvar=np.min(data[variable])
        maxvar=np.max(data[variable])
        
        #Calcula mínimo (si es negativo lo lleva a 0) y máximo
        if minvar<0:
            minvar=0

        min_x = np.argmin(data[variable])
        min_y = round(minvar,2)
        
        max_x = np.argmax(data[variable])
        max_y = round(maxvar,2)
        
        #Agrega el mínimo y el máximo a la gráfica
        ax1.scatter(data.index[min_x], min_y,c='k',label='minimo')
        ax1.scatter(data.index[max_x], max_y,c='r',label='maximo')

        ax1.annotate(f'{min_y}',xy=(min_x,min_y),xytext=(0,-20),textcoords="offset points",ha='center')
        ax1.annotate(f'{max_y}',xy=(max_x,max_y),xytext=(20,10),textcoords="offset points",ha='center')

        #Agrega el valor promedio en una línea puntuada
        ax1.plot(data.index,data[variable],'-',lw=1,label= variable,color=color)
        
        #Acomoda los ejes según el mínimo y el máximo
        ax1.set_ylim([(minvar)*0.99,(maxvar)*1.01])
        ax1.set_xlim([data.index[0],data.index[-1]])
        
        #Da formato a la fecha en el eje x
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M %n %m-%d'))
        
        ax1.tick_params(axis = 'y',color='k', labelcolor='k', labelsize=12)
        ax1.tick_params(axis = 'x',color='k', labelcolor='k', labelsize=12)
        
        ax1.set_title(variable.split()[0], fontsize=14, color='k')
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        
        ax1.set_ylabel(unit, fontsize=12, color='k')
        ax1.set_xlabel("Hora/Fecha", fontsize=12, color='k')

        toxic = {"CO [ppm]":9, 
                 "NO2 [ppb]": 53, 
                 "O3 [ppb]": 700, 
                 "NH3 [ppm]": 25, 
                 "PM01 [ug/m3]":30, 
                 "PM2_5 [ug/m3]":37, 
                 "PM10 [ug/m3]": 75}
        
        if variable in toxic.keys():
            plt.axhline(y=toxic[variable],color='k', lw=1.5, ls='--', label="umbral de toxicidad")

        plt.grid(False)
        
        plt.legend(loc=0,fontsize = 8)
        
        return fig

    def histogram(self,data,variable="CO [ppm]", unit = 'ppm',bins=10, color_hist ='#0E8388',
                shape_figs = (1,1),  position = [0,0], fig = plt.figure(figsize=(10,5))):
        """
        Graficar el histograma de la variable. Genera una grafica sencilla con el proposito de una visualizacion rapida de la variable.
        
        Argumentos:
        
        * ``data`` : DataFrame con los datos.
        * ``variable`` : Nombre de la variable de la unidad Simple a procesar.
        * ``unit`` : Unidad de la variable a procesar.
        * ``bins`` : Intervalos para el histograma.
        * ``kde``: Seleccionar si se graficara el kernel density estimate (KDE). Valor por defecto, True.
        * ``color hist`` : Color para el histograma.
        * ``color kde`` : Color para el kde.
        * ``labelsize`` :  Tamaño de fuentes   
        * ``shape_figs`` : shape de la matriz de figura en forma i,j. Por defecto (1,1)
        * ``position`` : Posición de la figura en la matriz en forma i,j. Por defecto [0,0]
        * ``fig`` : objeto de matplotlib
        """
        gs = gridspec.GridSpec(shape_figs[0],shape_figs[1])
        ax=fig.add_subplot(gs[position[0],position[1]])

        datos = data[variable]
        if self.check_variable(variable):
            plt.hist(datos.dropna(),bins = bins,density= True, label = variable, color=color_hist, edgecolor="white")
            sns.kdeplot(datos, alpha=0.8, linewidth=2, color='k')
            ax.axvline(np.percentile(datos.dropna(),50), color='#2E4F4F', 
                               ls="--", lw = 1.5, label = 'Mediana')
            ax.tick_params(axis = 'y',color='black', labelcolor='black', labelsize=12)
            ax.tick_params(axis = 'x', which = 'both',color='black', labelcolor='black', labelsize=12)
            ax.set_xlabel(f"{unit}", fontsize=12,  color='black')
            ax.set_ylabel("Densidad", fontsize=12, color='black')
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.set_title(variable, fontsize=14, color='k')
            plt.legend(loc= 0, fontsize = 8)

            return fig
            
        else:
            raise ValueError("Data Frame does not contain required variable")
        
    def fill_nan(self,method='hourly_mean'):
        """
        Reemplazar valores perdidos de todas las variables. 
        
        Argumentos:
        
        * ``method`` : Indica el metodo a utilizar para reemplazar los valores perdidos. ``hourly_mean`` para utilizar el promedio horario. ``mean`` para utlizar el promedio de todo el periodo.

        """
        if method=='hourly_mean':
            self.data.fillna(self.data.groupby(self.data.index.hour).transform('mean'),inplace=True)

        if method=="daily_mean":
            self.data.fillna(self.data.groupby(self.data.index.day).transform('mean'),inplace=True)
                
            
        elif method=='mean':
            self.data.fillna(self.data.mean(),inplace=True)

    def box_plot(self,variable=''):
        """
        Graficar el box plot de la variable indicada.
        
        Argumentos:
        
        * ``variable`` : Nombre de la variable de la unidad Simple a procesar.

        """
        if self.check_variable(variable):
            self.data.boxplot(column=variable)
        else:
            raise ValueError("Data Frame does not contain required variable")
                
    def outliers(self,variable='', method='z', fill_with='hourly_mean'):
        """
        Remover outliers presentes en la vaiable indicada. El valor de cada outlier sera reemplazado de acuerdo a la variable fill_with
        
        Argumentos:
        
        * ``variable`` : Nombre de la variable de la unidad Simple a procesar.
        * ``method`` : Metodo empleado para detectar los outliers. ``z``` para utlizar el metodo Z-score. ``IQR`` para utilizar el metodo interquartile range (IQR).
        * ``fill_with`` : Indica el metodo a utilizar para reemplazar los valores perdidos. ``hourly_mean`` para utilizar el promedio horario. ``mean`` para utlizar el promedio de todo el periodo.

        """
        if self.check_variable(variable):
            
                if method=='z':
                    data_mean, data_std = mean(self.data[variable]), std( self.data[variable])
                    # identify outliers
                    cut_off = data_std * 3
                    lower, upper = data_mean - cut_off, data_mean + cut_off
                    # identify outliers
                    
                elif method=='IQR':

                    q25, q75 = percentile(self.data[variable], 25), percentile(self.data[variable], 75)
                    iqr = q75 - q25
                    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
                    # calculate the outlier cutoff
                    cut_off = iqr * 1.5
                    lower, upper = q25 - cut_off, q75 + cut_off
                                        
                self.data[variable].values[self.data[variable].values>upper]=np.NaN
                self.data[variable].values[self.data[variable].values<lower]=np.NaN    
                self.fill_nan(method=fill_with) 
        else:
            raise ValueError("Data Frame does not contain required variable")

    def reindex_dataframe(self,datos):
    
        """
        Re-indexar el DataFrame para no obtener saltos en los registros.
        Los valores faltantes se reemplazan con Nan
        Argumentos:
        
        * ``datos`` : DataFrame de datos a reindexar.
        
        Retorna:
                
            * ``data_reindex`` :  DataFrame Re-indexado
        """
        
        data_reindex = pd.date_range(datos.index[0],datos.index[-1], freq = 'T')

        if len(data_reindex) == len(datos):
            print('Los datos están completos, no hay que reindexar')
            datos = datos
        else:
             print('Los datos están incompletos, hay que reindexar')
             print('Los datos faltantes son')
             for i in data_reindex:
                if i in datos.index:
                    pass
                else:
                    print(i)
        datos = datos.reindex(data_reindex)

        return datos

    def show_period(self,df=pd.DataFrame()):
    
        """
        Mostrar información sobre la calidad temporal de los registros.
        Se muestra la fecha inicial y final de medición, la frecuencia y el rango, 
        se cuantifican y eliminan datos duplicados, y se compara la cantidad de registros
        obtenidos por el sensor contra el máximo de registros que se puede tener según la 
        fecha inicial, final y la frecuencia de muestreo.
        
        * ``df`` : DataFrame de datos a procesar.
        
        Retorna:
                
            * ``df`` :  DataFrame sin datos duplicados
        """
        
        #Informacion de fechas
        fec_ini = df.index[0]
        fec_fin = df.index[-1]
        print('='*100)
        print(f"La fecha inicial del conjunto de datos es {fec_ini}")
        print(f"La fecha final del conjunto de datos es {fec_fin}")
        
        #Informacion de frecuencia
        time_diff = df.index.to_series().diff().dt.total_seconds().fillna(0) / 60
    
    # Create a new DataFrame to store the frequency of each index with the one before
        df_freq = pd.DataFrame(time_diff, columns=['Frequency'])
        df_freq_total = df_freq.mean()
        df_freq_total = round(df_freq_total)
        
        print('='*100)
        print(f"La frecuencia aproximada del registro de datos es de cada {df_freq_total} minutos.")

                 
        #Rango de datos
        rango = df.index[-1] - df.index[0]
        print(f"El rango de datos es de: {rango}")
        
        #Informacion de datos duplicados
        
        size_complete = len(df)
        df = df[~df.index.duplicated(keep='first')]
        duplicate = size_complete - len(df)
        if duplicate != 0:
            print('='*100)
            print(f'El dataset tiene en total {size_complete} registros de los cuales hay {duplicate} ' 
                  f'datos repetidos, \nes decir un '
                  f'{round((duplicate/size_complete)*100,2)}% del total de datos')
        else:
            print('='*100)
            print(f'El dataset tiene en total {size_complete} registros \nEl dataset no tiene datos repetidos')
            
        
        #Información de datos no registrados
        df_reindex = self.reindex_dataframe(df)
        count_nan = df.isnull().sum(axis = 0)
        
        print('='*100)
        print((f"De los {len(df)} registros válidos el {round((count_nan[count_nan.keys()[-1]]/len(df))*100,2)}% fueron NAN.\n"
               f"Además, teniendo en cuenta la fecha inicial y final y la frecuencia de los registros\n"
               f"El máximo posible de registro es de {len(df_reindex)}, en este sentido "
               f"se determina que el sensor \nregistró {round((len(df)/len(df_reindex))*100,2)}% " 
               f"del tiempo total"))
        
        return df_reindex 
    

    def define_period(self,initial='2021-01-01 00:00:00',final='2021-03-01 23:59:59'):
        """
        Definir un nuevo periodo para los datos de la unidad  Los valores que esten por fuera del nuevo periodo seran descartados. En caso que el inicio o el fin del
        nuevo periodo esten por fuera del periodo de datos actuales, seran reemplados por el inicio o fin del periodo actual, segun corresponda.
        
        Argumentos:
        
        * ``initial`` : Inicio del nuevo periodo. Formato %Y-%m-%d %H:%M:%S.%f. Por defecto 2021-01-01 00:00:00.0.
        * ``final`` : Fin del nuevo periodo. Formato %Y-%m-%d %H:%M:%S.%f. Por defecto 2021-03-01 23:59:59.99.
        """
        initial=datetime.strptime(initial,'%Y-%m-%d %H:%M:%S.%f')
        final=datetime.strptime(final,'%Y-%m-%d %H:%M:%S.%f')

        if self.data.index.name=='dates' or self.data.index.name=='time' or self.data.index.name=='Fecha_Hora':
            self.data.drop(self.data.index[self.data.index<initial],inplace=True)
            self.data.drop(self.data.index[self.data.index>final],inplace=True)
        else:
            print('There is no date variable in simple data')
        
        print('New Period')   
        self.show_data_period()   
    
    def resample(self,freq='T',method='mean',label='right'):
        """
        Remuestrear todas las variables con la nueva frecuencia freq.
        
        Argumentos:
        
        * ``freq`` : Nueva frequencia de muestreo. Ver formato frecuencia de pandas para referencia. Valor por defecto T (1 minuto).
        * ``method`` : Indica el metodo usado para calcular los valores correspondientes a la nueva frecuencia. mean para el promedio o sum para la suma.
        * ``label`` : Indica si el primer valor de la nueva frecuencia es el inicial (left) o el final (rigth)
        
        """
    
        if method=='mean':
            self.data = self.data.resample(freq, label=label).mean()
        elif method=='sum':
            self.data = self.data.resample(freq, label=label).sum()

    
    def include_variable(self,variable):
        """
        Incluye una nueva variable en el objeto DataManager. La nueva variable de ser un variable de un pandas Dataframe. El periodo de tiempo base del archivo no se 
        modificara con la nueva variable. Solo se incluiran los datos de la nueva variable que esten en el periodo de tiempo inicial del objeto DataManager.
         
        Argumentos:
        
        * ``variable`` : Nombre de la variable que se incluira en el objeto Datamanager.
        
        """
        name=self.data.index.name
        self.data = self.data.merge(variable, how="outer", left_index=True, right_index=True)
        self.data.index.name=name   

    def retrieve_data(fecha_i, fecha_f, id):
        """
        Llama los datos de la base de datos MongoDB y crea un dataframe.

        Argumentos:
        fecha_i: fecha inicial de muestreo, formato YYYY-MM-DD
        fecha_f: fecha final de muestreo, formato YYYY-MM-DD
        id: número de identificación de la unidad en la red (1-9)
            1. Poblado
            2. Bello
            3. EAFIT
            4. Llanogrande
            5. San Antonio de Prado
            6. Comuna 13
            7. Santo Domingo
            8. Copacabana
            9. Caldas

        Retorna:
        df: dataframe con índice tipo fecha
        """

        cluster = MongoClient("mongodb+srv://SimpleSpace:bio2343038@cluster0-xfp4r.gcp.mongodb.net/test?retryWrites=true&w=majority")
        database = cluster["test"]
        collection = database["s4"]

        result = collection.find({"date": {"$gte": str(fecha_i), "$lte": str(fecha_f)}, "simple_id": id}).sort("date",1)

        date = []
        pm01 = []
        pm2_5 = []
        pm10 = []
        co = []
        o3 = []
        nh3 = []
        nox = []
        no2 = []
        temp = []
        hum = []

        for i in result:
            pm01.append(i['pm01'])
            pm2_5.append(i['pm2_5'])
            pm10.append(i['pm10'])
            co.append(i['co'])
            o3.append(i['o3'])
            nh3.append(i['nh3'])
            nox.append(i['nox'])
            no2.append(i['no2'])
            temp.append(i['temperature'])
            hum.append(i['humidity'])
            date_str=(i["date"])
            time_str=(i['time'])
            combined = dt.datetime.strptime(date_str + " " + time_str, "%Y-%m-%d %H:%M:%S")
            date.append(combined)
        date = pd.to_datetime(date)
        date = sorted(date)
        
        df = pd.DataFrame(list(zip(pm01, pm2_5, pm10, co, o3, nh3, nox, no2, temp, hum)),
        columns=["PM01 [ug/m3]", "PM2_5 [ug/m3]", "PM10 [ug/m3]", "CO [ppm]", "O3 [ppb]", "NH3 [ppm]", 
                "NOx [ppb]", "NO2 [ppb]", "Temperatura [°C]", "Humedad [%RH]"], index=date)
        df.index.name = "Date" #convierte la fecha en el índice del dataframe

        return df

    def values_correction(self,data, method="offset"):
        """
        Corrección de los datos crudos. Los parámetros fueron calculados en referencia a los equipos SIATA y Vaisala en campañas de intercalibración.

        Argumentos:
        df: dataframe a corregir
        method: método de corrección (str). Puede ser "offset" o "rms", las dos estrategias de calibración aplicadas en Simple.
        Retorna:
        df_corr: dataframe corregido
        """

        #Valores shift y RMS para el método offset
        offset={"CO [ppm]": [-3.77, 0.755832860580499], 
                "O3 [ppb]": [-9.065490489642293, 0.6139909729834613],
                "NO2 [ppb]": [-12.777183333333372, 0.8402531047272038], 
                "Temperatura [°C]": [-8.13, 0.9986734918798236],
                "Humedad [%RH]": [8.75, 1.0541723382705244]
                }
        
        #Valores pendiente e intercepto para el método de regresión lineal
        rms={"CO [ppm]": [0.199389203722669, -0.5077818754493443], 
             "O3 [ppb]": [0.5669146779398607, -3.6024745326674754],
             "NO2 [ppb]": [0.7994489341472861, 6.919493105867987],
             "Temperatura [°C]": [0.39040137087822285, 13.034181911884504],
             "Humedad [%RH]": [1.2764418119200358, 1.3279097668697304]
        }

        df_corr = pd.DataFrame()
        df_corr.index = self.data.index

        #Las variables que se se corrigen con el método seleccionado.
        for var in ["PM01 [ug/m3]", "PM2_5 [ug/m3]", "PM10 [ug/m3]","NH3 [ppm]", "NOx [ppb]"]:
            df_corr[var] = self.data[var]

        #Las variables que se se corrigen con el método seleccionado.
        for var in ["CO [ppm]", "O3 [ppb]", "NO2 [ppb]", "Temperatura [°C]", "Humedad [%RH]"]:
            if method.lower() == "offset":
                df_corr[var] = (data[var]+offset[var][0])*offset[var][1]
            elif method.lower() == "rms":
                df_corr[var] = data[var]*rms[var][0]+rms[var][1]

        return df_corr


    def column_means(self,data="default"):
        """
        Calcula el promedio de las columnas del dataframe.

        Argumentos:
        dataframe

        Retorna:
        result_dict: lista de promedios según la columna del dataframe
        """
        result_dict = self.data.mean(axis=0, numeric_only=True).to_dict()
        
        return result_dict

    def calculate_aqi(concentration, pollutant_name):
        """
        Calcula el índice de calidad del aire (ICA o AQI en inglés) para un gas o material particulado.

        Argumentos:
        concentration: valor de concentración. Las unidades deben ser: CO [ppm], NO2 [ppb], O3 [ppb], PM2_5 [ug/m3] y PM10 [ug/m3].
        pollutant_name: nombre del contaminante: "CO", "NO2", "O3", "PM2_5", "PM10".

        Retorna: valor ICA del contaminante.
        """

        #Rangos de concentración para cada contaminante
        pollutant_ranges = {
            'CO': [(0, 4.49), (4.5, 9.49), (9.5, 12.49), (12.5, 15.49), (15.5, 30.49), (30.5, 40.49), (40.5, 50.4)],
            'NO2': [(0, 54.99), (55, 100.99), (101, 360.99), (361, 649.99), (650, 1249.99), (1250, 1649.99), (1650, 2049)],
            'O3': [(0, 54.99), (55, 70.99), (71, 84), (85, 104.99), (105, 200.99), (201, 404.99), (404, 504)],
            'PM2_5': [(0, 12.09), (12.1, 35.49), (35.5, 55.49), (55.5, 150.49), (150.5, 250.49), (250.5, 350.49), (350.5, 500.4)],
            'PM10': [(0, 54.99), (55, 154.99), (155, 254.99), (255, 354.99), (355, 424.99), (425, 504.99), (505, 604)]
        }
        # PM (ug/m3), gases (ppm, ppb)

        ranges = pollutant_ranges.get(pollutant_name, [])
        if not ranges:
            raise ValueError(f"Invalid pollutant name: {pollutant_name}")

        # Cálculo certificado del ICA, si no está en los rangos definidos, saca error.
        for i, (low, high) in enumerate(ranges):
            if low <= concentration <= high:
                index_low, index_high = i * 50, (i + 1) * 50
                break
        else:
            raise ValueError(f"Concentration value ({concentration}) outside the defined range.")
        
        aqi = ((index_high - index_low) / (high - low)) * (concentration - low) + index_low

        return aqi

    def overall_aqi(co_concentration, no2_concentration, o3_concentration, pm25_concentration, pm10_concentration):
        """
        Calcula el ICA total, calculando el máximo de los índices. Usa la función anterior.

        Argumentos:
        co_concentration: concentración de CO en ppm.
        no2_concentration: concentración de NO2 en ppb.
        o3_concentration: concentración de O3 en ppb.
        pm25_concentration: concentración de PM2_5 en ug/m3.
        pm10_concentration: concentración de PM10 en ug/m3.

        Retorna:
        overall_aqi: ICA total.
        """

        # concentrations=[co_concentration, no2_concentration, o3_concentration, pm25_concentration, pm10_concentration]

        # Convierte los valores NAN en 0.
        # concentrations[concentrations.values<0]=0
            
        
        #Calcula el ICA individualmente con la función calculate_aqi
        aqi_co = DataManager.calculate_aqi(co_concentration, 'CO')
        aqi_no2 = DataManager.calculate_aqi(no2_concentration, 'NO2')
        aqi_o3 = DataManager.calculate_aqi(o3_concentration, 'O3')
        aqi_pm25 = DataManager.calculate_aqi(pm25_concentration, 'PM2_5')
        aqi_pm10 = DataManager.calculate_aqi(pm10_concentration, 'PM10')

        overall_aqi_value = round(max(aqi_co, aqi_no2, aqi_o3, aqi_pm25, aqi_pm10),1)
        
        return overall_aqi_value

    def aqi_color(aqi):
        """
        Clasifica el ICA y define el color oficial.

        Argumentos:
        aqi: ICA

        Retorna:
        title, color: Nivel de norma, color que lo representa.
        """
        #Clasifica los valores del ICA con la clasificación y color representativo
        color_categories = {
            (0, 50): ('Bueno', 'lightgreen'),
            (51, 100): ('Moderado', 'yellow'),
            (101, 150): ('Dañino para grupos sensibles', 'orange'),
            (151, 200): ('Dañino', 'red'),
            (201, 300): ('Muy dañino', 'purple'),
            (301, float('inf')): ('Peligroso', 'darkpurple')
        }

        # Si no encuentra un rango al que pertenece el valor ingresado, lo considera como  Peligroso.
        for (low, high), (title, color) in color_categories.items():
            if low <= aqi <= high:
                return title, color
        return 'Peligroso','darkpurple'
