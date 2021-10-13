Para exportar la aplicación Python a un archivo ejecutable de Windows (.exe), ejecutar el siguiente comando:

    > pyinstaller app_diagnostico.spec

Requiere tener instalado pyinstaller en el entorno de desarrollo en el que se encuentre.

La exportación dará fallo en caso de no disponer de Nvidia CUDA 10.1 imstalado en el equipo. Esto se debe a que se incorporarán todos los ficheros binarios necesarios para que pueda emplearse la GPU.

Esta aplicación se puede encontrar exportada y comprimida a .7z en la siguiente dirección:

    https://drive.google.com/file/d/1booPeVAiYOCjxExoEEB56BkZVZZ0fT10/view?usp=sharing

