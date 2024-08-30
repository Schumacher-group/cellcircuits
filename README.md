### Cellcircuit Model
## Overview
The objective of this project is to simulate a cell circuit model to explain the emergence of scarring based on the work in a paper by [Adler et al.](https://www.cell.com/iscience/pdf/S2589-0042(20)30024-9.pdf) which focuses on macrophages (**M**) and myofibroblasts (**mF**) as main actors in an injured tissue.  
Both of hem secrete growth factors (PDGF and CSF) supporting the proliferation of the cell type -either in an autokrin or parakrin loop or even both. Furthermore, the growth factor binding is described by Michaelis-Menten functions and. Each differential equation is a composition of a proliferation and removal term where additionally myofibroblasts are assumed to have a maximal carrying capacity. This said, the equations of interest are:   

$$
\begin{align}
\dot{mF} &= mF \left( \lambda_1 \frac{PDGF}{k_1 + PDGF} \left( 1 - \frac{mF}{K} \right) - \mu_1 \right)  \\
    \dot{M} &= M \left( \lambda_2 \frac{CSF}{k_2 + CSF} - \mu_2 \right)  \\
    \dot{CSF} &= \beta_1 mF - \alpha_1 M \frac{CSF}{k_2 + CSF} - \gamma CSF  \\
    \dot{PDGF} &= \beta_2 M + \beta_3 mF - \alpha_2 mF \frac{PDGF}{k_1 + PDGF} - \gamma PDGF 
\end{align}
$$
To include injury signals, we model the signal as an influx $I(t)$ of macrophages and equation (2) becomes then
$$
\dot{M} = M \left( \lambda_2 \frac{CSF}{k_2 + CSF} - mu_2 \right) + I(t)
$$
For a in depth explanation of the derivation for the simulation and the parameter values, we refer to the attached documents.