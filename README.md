# Comparative study of the superpixel segmentation algorithm SLIC and Felzenszwalb


A segmentation in superpixels consists in dividing an image into groupings of pixels, called super-pixels. These superpixels should be close enough in size and shape, stick to the object's borders, or still have consistent colors.Superpixel segmentation is generally used as an image preprocessing step for computer vision, such as pattern recognition or even visual monitoring . Working on a pixel cluster rather than pixels has many advantages. On the one hand, by simplifying the representation of an image, it makes it possible to reduce the calculation time associated with the processing of these images. Another advantage is that superpixels allow an image to be observed at a larger scale, which provides better spatial information for measuring regional characteristics.

To create these superpixels, many algorithms have emerged in the literature (SEEDS, GMM, ERS, NCut, ...). These methods can be separated into two main classes : methods based on graphs and methods based on gradient. In order to have a global vision of superpixel segmentation, this thesis presents a gradient algorithm, the SLIC, then the superpixel segmentation algorithm of Felzenszwalb and Huttenlocher based on graph methods. These are two algorithms which have for objective the partitioning of the data but which will do it in a very different way, we can already see this difference in the form that the superpixels take in this figure.

![Screenshot](Data/slic_fh.png)

To be able to understand what are the advantages and the drawbacks of superpixel segmentation algorithms, it is important to study the properties of these algorithms. This report therefore also aims to propose a complete methodology for evaluating the performance of a superpixel segmentation algorithm. A comparative study will then be carried out on the two algorithms studied during this internship. It will make it possible to identify the strengths of this algorithms and the drawbacks. Although this methodology is performed on two superpixel segmentation algorithms, there are many others. This type of study is therefore part of the growing need to understand the benefits of each of these methods.

