Introducción
============

.. toctree::
   :maxdepth: 2

Es bien conocido experimentalmente que la materia pre-senta propiedades magnéticas. El magnetismo es una ciencia
con más de dos milenios de historia registrada. La atracción de los objetos ferrosos por un imán permanente a lo largo de
la distancia ha sido una fuente de curiosidad desde la edad del hierro. Las investigaciones de los fenómenos magnéticos
llevaron a la invención de los imanes de acero (agujas y herraduras) y la brújula que permitió la exploración del planeta.
El comportamiento concreto puede ser básicamente de tres tipos:

• Paramagnetismo: Una sustancia paramagnética tiene imanación nula para campo magnético externo nulo. Si el campo es no nulo, entonces se imanta en la misma dirección del campo. Así pues, la susceptibilidad magnética es positiva. Pueden considerarse dos clases de paramagnetismo: de Langevin o de Pauli.

    El paramagnetismo de Pauli es debido al gas de electrones libres de un metal. Requiere un tratamiento basado en las estadísticas cuánticas.

    El paramagnetismo de Langevin se debe a la interacción de los momentos magnéticos intrínsecos de las moléculas de un sólido con un campo magnético externo, y se estudia utilizando la estadística clásica, tanto discreta como continua.


• Diamagnetismo: Este caso es muy similar al anterior, excepto que la susceptibilidad magnética es negativa. La explicación del diamagnetismo es debida a Landau. El diamagnetismo tiene su origen en las órbitas circulares cuantizadas que describen los electrones libres de un metal. Requiere también el uso de la estadística cuántica.


• Ferromagnetismo: Es la propiedad que tiene una sustancia de presentar imanación espontánea a campo externo nulo por debajo de cierta temperatura crítica, llamada temperatura de Curie. Esta propiedad es debida a la interacción entre los momentos magnéticos intrínsecos de las moléculas de un sólido.

En este estudio haremos especial énfasis en las propiedades paramamagnéticas y ferromagnéticas de distintos sistemas.  Los fenómenos se puede analizar desde dos enfoques; el macroscópico y el microscópico. Para simplificar las definiciones y desarrollos, en adelante vamos a considerar la misma dirección para la magnetización M y el campo magnético H. Desde el punto de vista macroscópico, la ecuación de estado se expresa como:
:math:`M = M(H,T)`.

En  el  caso del paramagnetismo desde el punto de vista microscópico, las moléculas de un material  tienen momento magnético intrínseco no nulo, cuya magnitud se mide en unidades del magnetón de Bohr. Limitándonos a sólidos paramagnéticos no metálicos podemos considerar que dichas moléculas se encuentran localizadas en los nodos de una cierta red regular. Existen distintos modelos físicos que permiten estudiar el comportamiento de las propiedades físicas de los materiales, dentro de estos modelos se destacan el modelo de Ising y los modelos de Heisenberg cuántico y clásico.

Estos modelos, para el caso dos dimensional, toman al sistema como una red o matriz cuadrada en la que se ubican las partículas que componen al material. La energía del sistema se describe a través del siguiente Hamiltoniano:

:math:`\hat{H}=-\frac{1}{2}\sum_{<ij>} J_{ij} \hat{\sigma_i}\cdot \hat{\sigma_j} - \mu_B \sum_{i=1}^{N} \hat{\sigma}_i \cdot \vec{H}`

:math:`J` es la constante de intercambio, :math:`\hat{\sigma}_i` es el valor del spin para la i-ésima partícula en el arreglo y :math:`\vec{H}` es el vector de campo magnético externo al sistema. Para simular computacionalmente este tipo de modelos, se suelen usar "Métodos de Monte Carlo".