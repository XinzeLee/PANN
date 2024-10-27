# PANN: Physics-in-Architecture Neural Network for Power Electronics Modeling

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect--Xinze%20Li-blue)](https://www.linkedin.com/in/xinze-li-8199561b0/)
[![ORCID](https://img.shields.io/badge/ORCID-Xinze%20Li-brightgreen)](https://orcid.org/0000-0003-3513-209X)
[![GitHub](https://img.shields.io/badge/Github-XinzeLee-black?logo=github)](https://github.com/XinzeLee)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Xinze%20Li-cyan)](https://www.researchgate.net/scientific-contributions/Xinze-Li-2167307782)
[![Colab](https://img.shields.io/badge/Colab-PANN--Notebooks-red?logo=google-colab)](https://drive.google.com/drive/folders/1FXr82WQfBOj6xP01h-9RHIZBpiOFZwUC)
<br><br>
* Reference 1: <br>
X. Li et al., "Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network," in *IEEE Transactions on Industrial Electronics*, vol. 71, no. 11, pp. 14111-14123, Nov. 2024.<br>
[![DOI](https://img.shields.io/badge/DOI-10.1109/TIE.2024.3352119-cyan)](https://doi.org/10.1109/TIE.2024.3352119)
[![IEEE](https://img.shields.io/badge/IEEE-Xplore-orange)](https://ieeexplore.ieee.org/document/10463542)
[![ResearchGate](https://img.shields.io/badge/ResearchGate--1-blue)](https://www.researchgate.net/publication/378918445_Temporal_Modeling_for_Power_Converters_With_Physics-in-Architecture_Recurrent_Neural_Network)
* Reference 2: <br>
X. Li, F. Lin, X. Zhang, H. Ma and F. Blaabjerg, "Data-Light Physics-Informed Modeling for the Modulation Optimization of a Dual-Active-Bridge Converter," in *IEEE Transactions on Power Electronics*, vol. 39, no. 7, pp. 8770-8785, July 2024.<br>
[![DOI](https://img.shields.io/badge/DOI-10.1109/TPEL.2024.3378184-cyan)](https://doi.org/10.1109/TPEL.2024.3378184)
[![IEEE](https://img.shields.io/badge/IEEE-Xplore-orange)](https://ieeexplore.ieee.org/document/10473116)
[![ResearchGate](https://img.shields.io/badge/ResearchGate--2-blue)](https://www.researchgate.net/publication/379104054_Data-Light_Physics-Informed_Modeling_for_the_Modulation_Optimization_of_a_Dual-Active-Bridge_Converter)
* Reference 3: <br>
F. Lin, X. Li, X. Zhang and H. Ma, "STAR: One-Stop Optimization for Dual-Active-Bridge Converter With Robustness to Operational Diversity," in *IEEE Journal of Emerging and Selected Topics in Power Electronics*, vol. 12, no. 3, pp. 2758-2773, June 2024.<br>
[![DOI](https://img.shields.io/badge/DOI-10.1109/JESTPE.2024.3392684-cyan)](https://doi.org/10.1109/JESTPE.2024.3392684)
[![IEEE](https://img.shields.io/badge/IEEE-Xplore-orange)](https://ieeexplore.ieee.org/document/10506915)
[![ResearchGate](https://img.shields.io/badge/ResearchGate--3-blue)](https://www.researchgate.net/publication/380052824_STAR_One-Stop_Optimization_for_Dual_Active_Bridge_Converter_with_Robustness_to_Operational_Diversity)
* Reference 4: <br>
X. Li et al., "A Generic Modeling Approach for Dual-Active-Bridge Converter Family via Topology Transferrable Networks," in *IEEE Transactions on Industrial  Electronics*.<br>
[![DOI](https://img.shields.io/badge/DOI-10.1109/TIE.2024.3406858-cyan)](https://doi.org/10.1109/TIE.2024.3406858)
[![IEEE](https://img.shields.io/badge/IEEE-Xplore-orange)](https://ieeexplore.ieee.org/document/10627933)
[![ResearchGate](https://img.shields.io/badge/ResearchGate--4-blue)](https://www.researchgate.net/publication/382930411_A_Generic_Modeling_Approach_for_Dual-Active-Bridge_Converter_Family_via_Topology_Transferrable_Networks)
<br><br>
* Colabs:<br>
[![Colab](https://img.shields.io/badge/Colab-PANN--Buck-654062?logo=google-colab)](https://colab.research.google.com/drive/1FDxjR-LZxJBbp4PzsinhxdWMrUI7UjW-)
[![Colab](https://img.shields.io/badge/Colab-PANN--DAB-B4B4B3?logo=google-colab)](https://colab.research.google.com/drive/1dJ4GvKc03_eF__c8l1msbI7Fq-8a6ScD#scrollTo=2ede7f4b)
[![Colab](https://img.shields.io/badge/Colab-PANN--Operational--Diversity-26577C?logo=google-colab)](https://colab.research.google.com/drive/1PSpqhUEfGKXEfoSVesYUmhZCy4EpYTX9)
[![Colab](https://img.shields.io/badge/Colab-PANN--Topology--Transfer-E55604?logo=google-colab)](https://colab.research.google.com/drive/1jXo4uugvnRBgP2948HVPLNRsK8fCh-ge)
<br><br>

## Description
### I. PANN and its Structure
PANN, physics-in-architecture neural network, is a physics-informed neural network specifically for the modeling of power electronics systems, leveraging physics-crafted recurrent neural structure to embed discretized state-space circuit equations. PANN introduces inductive biases by embedding discretized PDEs directly into the network architecture, revealing data invariant directly. The neural architecture of PANN is shown in Fig. 1.
![Structure of PANN.](https://github.com/user-attachments/assets/af90a7b0-3e3e-4fad-bf8e-75bf7ce4efe3)
<br>Fig. 1. Structure of PANN.<br>
### II. PANN Inference
PANN inference is conducted by recurrently predicting the next state variables using the precomputed input variables and the inferred state variables from previous iteration. The PANN inference unfolded over time is given in Fig. 2.
![PANN Inference](https://github.com/user-attachments/assets/2c056085-9d77-4270-8c6e-fd3ed11ae78f)
<br>Fig. 2. Structure of PANN.<br>
### III. PANN's Explainability in Power Electronics
PANN model is explainable in power electronics, revealing circuit physical principles, switching behaviors, commutation loops, etc. Those power electronics insights discovered by PANN for an exemplary non-resonant Dual-Active-Bridge converter are shown in Fig. 3.
![PANN's Explainability](https://github.com/user-attachments/assets/57593884-9546-4964-9c5a-b8926376df86)
<br>Fig. 3. PANN's physical explainability.<br>
### IV. PANN Training
The training workflow of PANN is shown in Fig. 4, and one training epoch for PANN is shown in Fig. 5.
![PANN's training workflow](https://github.com/user-attachments/assets/84258774-3626-46d8-8bfc-27befe24256a)
<br>Fig. 4. PANN's training workflow.<br><br>
![One training epoch for PANN](https://github.com/user-attachments/assets/c70eb196-d688-4468-be95-5d23e8639ae1)
<br>Fig. 5. One training epoch for PANN.<br>
### V. PANN is Data-Light and Lightweight
The PANN is data-light, as it directly embeds circuit physical principles into its neural architecutre, ensuring stringent physical consistency. PANN requires only few data samples for training, reducing data requirements by 3 orders of magnitude. Theoretically, PANN only requires a dataset containing time-series points no less than the number of defined converter parameters. Additionally, PANN is similar to a single-layer recurrent neural network with only a few neural parameters, so it exhibits lightweight advantage. The light advantages of PANN are illustrated in Fig. 6. 
![PANN is light AI model](https://github.com/user-attachments/assets/1b5a4366-8fd8-401a-b2cd-b9a7708b5e6f)
<br>Fig. 6. Data-light and lightweight advantages of PANN.<br>
### VI. PANN is Flexible
The PANN is flexible in terms of four main aspects: operating conditions, modulation strategy, performance metrics, and circuit parameters and topological variants, as summarized in Fig. 7.
![PANN is flexible](https://github.com/user-attachments/assets/6aededa3-b539-4d1c-90b7-d217ffbf213f)
<br>Fig. 7. Flexibility of PANN.<br>
### VII. Customize PANN to your Application/Converter of Interests
The steps shown in Fig. 8 can be followed to customize PANN to your specific application or converter of interests. Fig. 8 showcases the modeling of non-resonant DAB converters.
![PANN for DAB](https://github.com/user-attachments/assets/da096d81-b48f-41e4-84df-7c2d604821c6)
<br>Fig. 8. Case study: Design PANN for DAB converters.<br>
<br><br>

## PANN Tutorial
A comprehensive tutorial of PANN is given in [PANN_Tutorial.pdf](./tutorials/PANN_Tutorial.pdf), with the main topic of "***The Next Generation of AI for Power Electronics: Explainable, Light, and Flexible***". The slides are prepared by Xinze Li and Fanfan Lin. <br><br>
It covers basic introduction to applications of AI in power electronics, brief discussion of physics-informed machine learning methods, PANN inference and its explainability, PANN training and its light characteristics, and PANN's flexibility across diverse conditions and topologies (out-of-domain transfer capability). 
<br><br>


## Deploy PANN
To deploy PE-GPT on your PC, the first step is to setup your API call to OpenAI models, please see core/llm/llm.py for more details. <br>
If you want to interact with Plecs software to simulate the designed modulation for DAB, you need to enable the xml-rpc interface in Plecs settings.

```bash
# clone the github repository
git clone https://github.com/XinzeLee/PANN

# change the current working directory
cd PANN

# install all required dependencies
pip install -r requirements.txt

# Now you can import the customized PANN models defined.
# It is recommended to go through the notebooks first, before you start to implement on your own.
```
<br><br>

## Run Notebooks on Google Colab
Although it is strongly recommended to try out those notebooks on your local machine (as the graphical plots will be easier to view and play with), We have still added a few Google Colab notebooks. 
- [Google Colab (Pytorch) PANN-Buck](https://colab.research.google.com/drive/1FDxjR-LZxJBbp4PzsinhxdWMrUI7UjW-)
- [Google Colab (Pytorch) PANN-DAB](https://colab.research.google.com/drive/1dJ4GvKc03_eF__c8l1msbI7Fq-8a6ScD#scrollTo=2ede7f4b)
- [Google Colab (Pytorch) PANN-Operational-Diversity](https://colab.research.google.com/drive/1PSpqhUEfGKXEfoSVesYUmhZCy4EpYTX9)
- [Google Colab (Pytorch) PANN-Topology-Transfer](https://colab.research.google.com/drive/1jXo4uugvnRBgP2948HVPLNRsK8fCh-ge)
<br><br>

## Code Author
@code-author: <br>
* Xinze Li (email: xinzeli831@gmail.com)
* Fanfan Lin (email: fanfanlin31@gmail.com)

<br><br>
## Notes
This repository provides a simplified version of the PE-GPT methodology presented in our journal paper. Despite the simplifications, the released code preserves the overall core architecture of the proposed PE-GPT.
<br>
This repository currently includes the following functions/blocks: Retrieval augmented generation, LLM agents, Model Zoo (with a physics-in-architecture neural network deployed in ONNX engine for modeling DAB converters), metaheuristic algorithm for optimization, simulation verification, graphical user interface, and knowledge base. Please note that the current knowledge base is a simplified version for illustration. 

<br><br>
## License

This code is licensed under the [Apache License Version 2.0](./LICENSE).
