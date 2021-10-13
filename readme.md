# Detección de signos de Retinopatía Diabética en imágenes de fondo de ojo

Aplicación de un sistema de clasificación de imágenes basado en Deep Learning a la detección de <b>Retinopatía Diabética (RD)</b> en imágenes de fondo de ojo. Esta enfermerdad posee cinco niveles de gravedad, que son:

- <b>0</b>: sin RD aparente (sano)
- <b>1</b>: RD no proliferante leve
- <b>2</b>: RD no proliferante moderada
- <b>3</b>: RD no proliferante severa
- <b>4</b>: RD proliferante

Debido a ello, se proponen dos redes con la misma arquitectura (EfficientNetB5). La primera red clasifica las imágenes en dos clases, '0' y '1', siendo '1' que la imagen contiene algún indicio de RD (nivel 1 o superior). La segunda red distingue las imágenes en tres clases, '0', '1' y '2', siendo '1' RD no proliferante leve y '2', los estados más avanzados de la enfermedad.

El entrenamiento de ambas redes se ha realizado empleando el dataset de EYEPACS de más de 88.000 imágenes etiquetadas para RD, disponible públicamente en <a href='https://www.kaggle.com/c/diabetic-retinopathy-detection/data'>Kaggle</a>. Para el testeo, se ha recurrido, además, a otro dataset frecuentemente mencionado en la literatura, <a href='https://www.adcis.net/en/third-party/messidor2/'>Messidor-2</a>, formado por 1.748 imágenes.

A las imágenes de entrenamiento se les aplica un procedimiento de data-augmentation que efecturará UNO de las siguientes transformaciones a cada imagen analizada:
- Adición de ruido, el cual puede ser uniforme, de Laplace o de Poisson.
- Volteo sobre el eje Y.
- Volteo sobre el eje X.
- Rotación de la imagen, que será hacia la izquierda o hacia la derecha.
- Transformación del color, realizado mediante un desplazamiento del canal H en el espacio de color HUE.

Además, independientemente de cualquier transformación de las que se han citado, en un 80% de las veces se aplicará una operación de jigsaw.

Tras finalizar los entrenamientos de ambas redes, se evaluó su rendimiento en el conjunto de validación (compuesto por imágenes de EYEPACS) y se extrajeron los umbrales que alcanzaban un 90% de sensibilidad en la predicción de casos de RD (nivel 1 o superior). Estos umbrales fueron trasladados a los conjuntos de test, obteniendo, para el dataset Messidor-2, los siguientes resultados:

- <u>Primera red</u>: 95.45% de sensibilidad y 54.58% de especificidad
- <u>Segunda red</u>: 97.63% de sensibilidad y 47.22% de especificidad

Una descripción más detallada de los resultados obtenidos y la distribución de los datasets puede ser consultada en los respectivos informes de resultados, en formato pdf.

Las implementaciones de los entrenamientos de estas dos redes se encuentran en los ficheros: prueba19_MPropio_EfficientNetB5_SGB_EYEPACS_Bal.ipynb y prueba20_Modelo_prop_EfficientNetB5_SGD_EYEPACS_Bal_0-1-234.py.

### <u>Requisitos</u>

Los módulos necesarios para ejecutar el proyecto son los que se indican a continuación:

<font face='Courier New'>

- python == 3.7.9
- tensorflow-gpu == 2.4
- open-cv == 4.4.0
- pandas == 1.1.4
- numpy == 1.18.3
- seaborn == 0.11.2
- matplotlib == 3.4.2
- scikit-learn == 0.23.2
- imgaug == 0.4.0
- pillow (PIL) == 8.0.1
- xlrd == 2.0
- pyqt5 (para la aplicación gráfica)

</font>

## <u>Distribución del proyecto</u>

El proyecto se encuentra estructurado en las siguientes carpetas:

### <u>data</u>

En este directorio se podrán encontrar todos los datasets que han sido descargados y/o utilizados en el entrenamiento y testeo de las redes. Dentro se encuentran tres directorios:

- <u>downloaded_datasets</u>: contiene los archivos comprimidos de los datasets tal y como se obtuvieron al descargarse desde sus respectivas páginas.
- <u>original_datasets</u>: contiene todas las imágenes y los etiquetados propios de cada dataset. En estos directorios se generarán y guardarán un fichero .csv por cada dataset que contendrá toda la información necesaria por cada imagen.
- <u>processed_datasets</u>: en esta carpeta, se encontrarán las imágenes de los datasets repartidas de una forma más amigable con el usuario. Además, estas imágenes ya se encontrarán recortadas y redimensionadas al tamaño deseado. Con estas imágenes se procederá a entrenar y testear las redes.

Una descripción más detallada de la estructura de los datasets y los ficheros .csv se puede encontrar en el fichero readme.txt dentro del directorio data, y en los Infomes de resultados.

### <u>lib</u>

Directorio donde se almacenan los módulos Python implementados para el desarrollo del proyecto. Éstos son:

- <u>custom_callbacks</u>: en este módulo se encuentran los Callbacks y funciones que han sido necesarios implementar para realizar los entrenamientos. 
- <u>custom_metrics</u>: contiene las diferentes métricas implementadas para efectuar un correcto seguimiento de los entrenamientos.
- <u>data_augmentation</u>: este módulo contiene funciones útiles para la aplicación de procedimientos de data-augmentation. Incluye además la definición de los objetos 'augmenters' de la librería imgaug con los que se realizará las operaciones de data augmentation. Concretamente, el objeto que contiene la definición del proceso de data-augmentation se llama 'aug'.
- <u>dataset</u>: contiene las funciones de mapeado necesarias para la carga de los datasets en memoria. El proceso de carga de un dataset es realizado por la función 'create_dataset_new', configurable a través de sus parámetros. A descatar que esta función espera recibir las imágenes del dataset mediante un fichero .csv que contenga la ruta de la imagen, la etiqueta asociada y su nivel de RD (este fichero se genera mediante el script a03_define_custom_dataset.py).
- <u>evaluation</u>: este módulo contiene las distintas funciones que han sido programadas y utilizadas para la evaluación de las redes una vez han sido entrenadas. Principalmente son funciones enfocadas en el cálculo de valores a partir de las ROCs.
- <u>plotting</u>: contiene las funciones encargadas de mostrar gráficamente los resultados obtenidos por las funciones del módulo anterior (ROCs y matrices de confusión).
- <u>preprocess</u>: en este módulo se definen las funciones empleadas durante la etapa inicial de las imágenes, la detección y recorte de la circunferencia de la retina.
- <u>models</u>: este último módulo se trata en realidad de otro directorio en el que dentro se definen las diferentes arquitecturas probadas. Por cada arquitectura, habrá un archivo Python que contendrá la función 'get_model', que recibirá por parámetro el tamaño de la imagen de entrada esperada y el número de salidas que tendrá la red.

### <u>logs</u>

Este directorio fue creado por el Callback de Tensorboard y contiene los datos que necesita para poder representar gráficamente la evolución de los distintos entrenamientos en los que se usó. 

Este Callback se dejó de usar dada su ineficiencia en tiempo de ejecución. En su lugar, el seguimiento de los últimos entrenamientos ha sido llevado a cabo mediante el Callback propio 'Save_Training_Evolution', el cual genera un fichero csv con todos los valores de cada métrica al final de cada época (se almacenará en la carpeta correspondiente de la red, en el directorio 'saved_weights'). Estos ficheros se podrán visualizar y/o comparar en cualquier momento con el script 'see_metrics_evolution_in_training', ubicado en el directorio raiz del proyecto.

### <u>aplicacion</u>

En este directorio se encuentra la implementación de la aplicación gráfica (app_diagnóstico) desarrollada para la predicción de RD en imágenes. Toma como base la implementación de un visualizador de imágenes hecho en PyQt5 (<a href="https://github.com/baoboa/pyqt5/blob/master/examples/widgets/imageviewer.py">Image Viewer</a>), sobre el cual se han realizado las modificaciones necesarias para que pueda efectuar la tarea de diagnóstico.

Se incluye un pequeño manual de uso de la aplicación en formato pdf.

Una versión exportada a ejecutable nativo de Windows de esta aplicación puede ser encontrada en: https://drive.google.com/file/d/1booPeVAiYOCjxExoEEB56BkZVZZ0fT10/view?usp=sharing 

### <u>saved_weights</u>

Directorio en el que se almacenan las configuraciones de pesos de las redes entrenadas. Por cada arquitectura de red hay un directorio, y dentro de éstos, una serie de directorios nombrados de forma acorde a los parámetros establecidos para esos entrenamientos.

Las dos redes propuestas en este proyecto se encuentran en el directorio correspondiente su arquitectura: efficientNetB5. 

Dentro, los datos del entrenamiento de la red diseñada para predecir 2 clases (sanos frente a cualquier nivel de RD) se encuentran en el directorio 'SGD_bal_bs4'. Por otro lado, la red con 3 salidas (sanos, indicios de RD leve, indicios de RD moderada o peor) se encuentra en la carpeta 'SGD_bal-0-1-234_bs4'.

Además, en estas carpetas (las de las dos redes seleccionadas) se encuentran las salidas que las redes generaron para el dataset de validación en cada una de las etapas de validación del entrenamiento. También se encuentran las predicciones obtenidas para los distintos datasets de test definidos y los falsos negativos.

### <u>entrenamientos_prueba</u>

En este directorio se han recogido los scripts y notebooks desarrollados previamente en los que se definieron otros entrenamientos que no llegaron a mostrar el comportamiento deseado. La causa principal fue una serie de pequeños errores en la carga de los datasets y la forma en que el conjunto de imágenes de entrenamiento se mezclaban. En caso de necesitar ejecutar alguno de estos scripts o notebooks, éste deberá ser copiado antes a la ruta raiz del proyecto para evitar problemas de importación de librerías.

El archivo pdf 'historial_pruebas' recoge los resultados de las diversas pruebas realizadas.

Incluye además una copia del dataset de dígitos de MNIST, utilizado para testear el procedimiento de carga de las imágenes.


## <u>Scripts, notebooks y archivos</u>

Además de las carpetas mencionadas donde se recogen los distintos módulos implementados y los datos generados, se implementaron una serie de scripts y notebooks que hacen uso de estos elementos.

### <u>scripts de manejo de los datasets</u>

Estos scripts, que mantienen una numeración con formato 'aXX_', realizan todo el procesamiento previo al entrenamiento, la preparación de los datasets.

#### <u>a00_fix_eyepacs_zip_files.sh</u>
Este script toma todos los ficheros comprimidos del dataset EYEPACS, los cuales se encuentran erróneamente particionados por defecto, y los corrige para poder realizar su extracción. Se espera que estos ficheros se encuentren en la ruta: 'data/downloaded_datasets/eyepacs/downloads/'.

#### <u>scripts formato a01_...</u>
Estos scripts (habrá uno por cada dataset descomprimido) realizarán la tarea de agrupar en un fichero .csv general toda la información relevante de las imágenes del dataset en cuestión. Leerá los diferentes archivos que tenga el dataset para etiquetar sus imágenes y, a partir de ello, confeccionará un nuevo fichero .csv. El dataset deberá encontrarse almacenado en 'data/original_datasets/', donde previamente habrá sido descomprimido de forma manual.

El fichero .csv generado tendrá la siguiente estructura para todos los datasets:

<table style="width:100%"><tr>
<th>image</th>
<th>path</th>
<th>DR_level</th>
<th>DME_level</th>
<th>gradability</th>
<th>size_x</th>
<th>size_y</th>
<th>cntr_radius</th>
<th>cntr_center_x</th>
<th>cntr_center_y</th>
</tr></table>

El significado de cada columna es el siguiente:
- <u>image</u>: nombre del archivo de la imagen, sin extensión.
- <u>path</u>: ruta completa a la imagen, comenzando desde el directorio 'data/'. Por ejemplo: 'data/original_datasets/eyepacs/10_left.jpeg'.
- <u>DR_level</u>: es el nivel de RD diagnosticado para la imagen seleccionada. Puede tomar valores desde 0 hasta 4.
- <u>DME_level</u>: indica la presencia (1) o no (0) de DME en la imagen. 
- <u>gradablility</u>: indica si la imagen seleccionada tiene calidad suficiente para realizar un buen diagnóstico o no (0 - insuficiente calidad, 1 - buena calidad).
- <u>size_x</u>: ancho de la imagen, en píxeles.
- <u>size_y</u>: alto de la imagen, en píxeles.
- <u>cntr_radius</u>: radio del la circunferencia detectada presumiblemente de la retina.
- <u>cntr_center_x</u>: coordenada x del centro de la circunferencia derectada en la imagen.
- <u>cntr_center_y</u>: coordenada y del centro de la circunferencia derectada en la imagen.

Por último, en caso de que algún dataset no disponga del etiquetado para alguno de los campos mencionados, se asignará un valor -1. Por tanto, -1 representará 'valor desconocido'. Y en caso de que se desconozca por completo el diagnóstico de una imagen, se etiquetará a ésta como imagen sin calidad (gradability=0).

De esta forma, se facilitan posteriores procesos, ya que todos los datasets estarán estructurados de idéntica manera.

#### <u>Script a02_redistribute_datasets.py</u>

Este script toma los datasets que reciba como parámetros y creará por cada uno de ellos un nuevo directorio en 'data/processed_datasets/'. Por cada directorio principal del dataset, se definirán:
- Un directorio que recogerá todas aquellas imágenes pertenecientes al dataset seleccionado que fueron etiquetadas como imágenes con insuficiente calidad.
- Un directorio por cada nivel de RD existente en el dataset.
- Un directorio para aquellas imágenes a las que no se les pudo detectar correctamente la circunferencia de la retina.

En caso de que alguna de las carpetas quedase vacía, ésta no se creará.

Las imágenes serán repartidas entre las distintas carpetas previamente mencionadas una vez hayan sido procesadas mediante el recorte y centrado del área de interés (la circunferencia de la retina) y su redimensionado al tamaño deseado (en este caso, 540x540). Las nuevas imágenes tendrán formato .png.

Posteriormente, se generará un nuevo fichero .csv por cada dataset, almacenado en su nueva carpeta en 'data/processed_datasets/', que tendrá la siguiente estructura:

<table style="width:100%"><tr>
<th>image</th>
<th>path</th>
<th>DR_level</th>
<th>DME_level</th>
<th>gradability</th>
<th>old_size_x</th>
<th>old_size_y</th>
</tr></table>

Donde las columnas significan:
- <u>image</u>: nombre del archivo de la imagen procesada, sin extensión.
- <u>path</u>: ruta completa a la imagen procesada, partiendo desde 'data/processed_datasets/'.
- <u>DR_level</u>: nivel de RD diagnosticado.
- <u>DME_level</u>: presencia o no de DME.
- <u>gradability</u>: indicativo de la calidad de la imagen para diagnosticar.
- <u>old_size_x</u>: ancho de la imagen original previa al recorte.
- <u>old_size_y</u>: altura de la imagen original previa al recorte.

En este caso, deja de ser necesario indicar la posición de la circunferencia en las imágenes puesto que éstas ya han sido recortadas. Sí se conserva el tamaño original de éstas en caso de ser necesario descartar aquellas que han sufrido un redimensionado de gran factor de escala y no posean suficiente calidad.

Nuevamente, en caso de no conocer el valor de un atributo de la imagen, éste tendrá asignado un -1.

Más información acerca de los parámetros definidos para este script y su uso se puede encontrar al ejecutar:

    python a02_redistribute_datasets.py -h

#### <u>Script a03_define_custom_dataset.py</u>

Este script permitirá definir datasets de entrenamiento, validación y test mediante la mezcla de los datasets existentes, que deberán haber sido previamente procesados por el script a02_redistribute_datasets.py (y almacenados en 'data/processed_datasets/').

A través de distintos parámetros, se podrá establecer en cuantas clases se quieren agrupar las imágenes, qué proporcion se debe seguir en su división en los subconjuntos de entrenamiento, validación y test, si utilizar solamente imágenes con buena calidad...

Para más información acerca de los parámetros de este script, consultar su descripción inicial o ejecutar:

    python a03_define_custom_dataset.py -h

#### <u>Script a04_create_copy_of_datasets.py</u>

Este script permite crear una copia de todas las imágenes recogidas en un dataset dado (fichero .csv que previamente habrá sido generado por el script anterior) y crear un nuevo fichero .csv que recoja esa nueva ruta hacia las imágenes.

Es un script útil en caso de necesitar crear una pequeña copia del dataset que poder enviar a otro equipo.

#### <u>Script a05_balance_dataset.py</u>

Este script permite crear datasets balanceados a partir de un dataset dado (fichero .csv que previamente habrá sido generado por el script a03_define_custom_dataset). 

La operación de balanceo se realizará a partir de cómo se reparten los distintos niveles de RD en las clases que el usuario defina, por ejemplo 0 y 1234.

Para más información acerca de los parámetros de este script, consultar su descripción inicial o ejecutar:

    python a05_balance_dataset.py -h

### <u>Archivos de datasets e información</u>

Se incluyen en el proyecto una serie de archivos en formato .csv que representan los distintos datasets de entrenamiento, validación (balanceado y sin balancear) y test que han sido utilizados en el desarrollo del trabajo.

También se incluyen otros archivos de texto o pdf que explican con más detalle algunos elementos aquí mencionados como los resultados o la distribución de los datasets.

### <u>Script see_metrics_evolution_in_training.py</u>

Este script, mencionado anteriormente, permite visualizar y/o comparar la evolución que muestra un entrenamiento, ya sea mientras éste se ejecuta o una vez finalizado, mediante la consulta del fichero .csv que se va generando conforme el entrenamiento progresa (para que se genere dicho .csv es necesario que en el entrenamiento se incluya el Callback Save_Training_Evolution).

Para más información acerca de los parámetros de este script, consultar su descripción inicial o ejecutar:

    python see_metrics_evolution_in_training.py -h

### <u>Notebook nb01_data_augmentation.ipynb</u>

Este notebook fue empleado para visualizar las diferentes transformaciones disponibles en la librería imgaug y, a partir de ahí, definir un modelo de data-augmentation que genere imágenes diferentes pero realistas.

### <u>Archivos prueba19 y prueba20</u>

Estos dos archivos, notebook y script, contienen la definición del entrenamiendo de las dos redes propuestas. En ambos casos:
- Se construyen unos datasets de entrenamiento y validación balanceados (validación está balanceado ya en el fichero .csv), estableciendo un <i>batch size</i> de 4 para entrenamiento y 12 para validación.
- Se emplea SGD como optimizador del entrenamiento, indicando un <i>learning rate</i> de 0,0001, <i>momentum</i> de 0,9 y un <i>clipnorm</i> igual a 1. En el entrenamiento de la red con 3 salidas (0, 1 y 2), se aplica un descenso del 10% al <i>learning rate</i> cada 10 épocas.
- Se establece una duración máxima de 2.000 épocas, aunque son detenidos de forma manual una vez se visualizó gráficamente su estancamiento o empezó a mostrar indicios de <i>overfitting</i>.

### <u>Notebook nb04_testing_models.ipynb</u>

Este notebook contiene el código encargado de evaluar ambas redes, mostrando para cada una de ellas la ROC que obtienen en cada dataset y el AUC calculado. Además, extrae los umbrales que garantizan unos niveles mínimos de sensibilidad en el dataset de validación que posteriormente se aplican a los datasets de test.

También permite consultar las tasas de falsos negativos obtenidos en cada dataset tanto de forma númerica como a través de matrices de confusión.