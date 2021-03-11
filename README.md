# locality-sensitive-hashing
Report and corresponding Python code for an assignment on locality sensitive hashingfor the course Advances in Data Mining at Leiden University. The project has been carried out together with Freek van Geffen.

Three documents are included in this repository:

1. The actual assignment
2. A written report
3. Python code

Results presented in report.pdf can be reproduced by running the main.py file. 

In order to get the file running, you should have the following file in the same (sub)folder as the main.py file:
	- user_movie_rating.npy
This file can be requested by sending an e-mail to justinkraaijenbrink@outlook.com.

Use the following command line: python main.py -d /very/long/path/to/user_movie_rating.npy -s seed -m js/cs/dcs
Where seed should be an integer and one can choice to use js (Jaccard Similarity), cs (Cosine Similarity) or dcs (Discrete Cosine Similarity)

Best regards,
Freek and Justin
