# VAQEM

VAQEM incorporates error mitigation techniques (those that have tunable parameters) into the variational tuning harness of VQA algorithms, i.e. the error mitigation parameters are tuned along with the ansatz rotational gate angles.

The example code in this repo (at the moment) performs VAQEM for VQE + Dynamic Decoupling. The number of DD instances in idle windows is being tuned via the vaqem approach.  This is shown for a simple 6qubit TFIM VQE ansatz (but this is paramterized and can be easily changed). The tuner uses imfil from skquant but this can be changed too (but some tuners are super slow). The example code is with noise simulation, please read note on noise sim below.


IMPORTANT - we observe that decoherence mitigation is much more beneficial on the real device than in simulation, due to apparent simulator limitations on how noise is modeled / mitigated. therefore for simulation, noise needs to be scaled up for observable results and can be used for sanity checks. real machine result (which ofc has no scaling etc) are the only things that really matter. all our results in the paper are from real machines. 1q gate rescheduling is especially not seen to be useful in sim and needs to be real machine tested. in the paper's discussion section, we discuss how simulator and real machine act very differently. 

Converting this code to support real machine runs is non trivial since that will require: a) either being able to perform the below within qiskit runtime (not possible in 2021) or b) running one iteration at a time on the real mahine which is slow. plus this requires the below code to be broken into two parts, the first part creates the circuit and runs it on the machine, the 2nd part extracts results, performs compuations, and readies next iteration

TODO: I hope to add code for real machine runs soon once i get it verified on the newer qiskit / quantum devices.

TODO: it would be worthwhile integrating VAQEM with Qiskit's own DD.





Tested on
```
{'qiskit-terra': '0.21.2', 'qiskit-aer': '0.11.0', 'qiskit-ignis': '0.2.0', 'qiskit-ibmq-provider': '0.19.2', 'qiskit': '0.38.0', 'qiskit-nature': None, 'qiskit-finance': None, 'qiskit-optimization': '0.4.0', 'qiskit-machine-learning': None}
```



Paper: https://arxiv.org/abs/2112.05821

Bibtex:
```
@misc{https://doi.org/10.48550/arxiv.2112.05821,
  doi = {10.48550/ARXIV.2112.05821},
  
  url = {https://arxiv.org/abs/2112.05821},
  
  author = {Ravi, Gokul Subramanian and Smith, Kaitlin N. and Gokhale, Pranav and Mari, Andrea and Earnest, Nathan and Javadi-Abhari, Ali and Chong, Frederic T.},
  
  keywords = {Quantum Physics (quant-ph), FOS: Physical sciences, FOS: Physical sciences},
  
  title = {VAQEM: A Variational Approach to Quantum Error Mitigation},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
