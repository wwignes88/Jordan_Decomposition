# Jordan_Decomposition


The READ_ME2 pdf file corresponds to the original posted version which is the Jordan_decomposition.py file which uses the files contained in the funcs_ folder. The derivations and example presented there follows  ref. [1] below. In hindsight it was perhaps not the best source to follow as she was a graduate student seemingly trying to demonstrate the depth of her knowledge and, in a fashion typical of graduate students, likely trying to make sense of things as she was writing it. Why she performs two separate transformations (null-rank then eigen) is unclear. Nevertheless, she did a good job and so I wrote the code and write-up based on her paper. The code is HORRENDOUSLY innefficient and will not work for many matrices. It does however, work for the example presented in her paper and it provides detailed printouts of what is happening -- a thing that may prove of use to those looking to learn about Jordan decomposition.

The updated version has all the neccessary files contained in one folder. This version follows the steps presented in 'Linear Algebra Done Right' by Axler[2] (pgs. 271-273 and  procedure 2.31) for calculating the Jordan transformation basis. I HIGHLY reccomend this book. The procedure for finding the minimum polynomial remains mostly unchanged from the original version, so the reader can consult the READ_ME2 pdf file for an explanation of this process.



REFERENCES:

[1]. 'The Jordan canonical form' by Attila M´at´e., Brooklyn College of the City University of New York, 2014.

[2] 'Linear Algebra Done Right' [Third Edition] by Sheldon, Axler. Springer, 2015.
