# FqClustering

Python scripts for clustering neurons from morphological reconstructions using form factor F(q) curves.
The methodology has been applieed to three different morphological reconstruction datasets including [mouse primary visual cortex (V1)](https://www.nature.com/articles/s41593-019-0417-0) from the [Allen Cell Type database](https://celltypes.brain-map.org/data), [*Drosophila* olfactory projection neurons (PN)](https://www.cell.com/cell/fulltext/S0092-8674(18)30787-6) from the [TEMCA2 dataset](https://github.com/bocklab/temca2data/tree/master/geometry_analysis/data), and the [*C. elegans* nervous system](https://royalsocietypublishing.org/doi/10.1098/rstb.2017.0382) from the [Openworm c302 framework](https://github.com/openworm/c302).

- ALLEN_fq_calc.py: the Python script for calculating F(q) curves from mouse primary visual cortex (V1) reconstructions
- ALLEN_fq_cluster.py: the main Python script for the data analysis and figure reproduction for mouse primary visual cortex (V1)
- dendrite_type.npy: .npy file containing dendrite type labels for the neurons in mouse primary visual cortex (V1) queried from the Allen Cell Type database
- CE_fq_calc.py: the Python script for calculating F(q) curves from the *C. elegans* nervous system
- CE_fq_cluster.py: the main Python script for the data analysis and figure reproduction for the *C. elegans* nervous system
- Dros_fq_calc.py: the Python script for calculating F(q) curves from *Drosophila* olfactory projection neurons (PN)
- Dros_fq_cluster.py: the main Python script for the data analysis and figure reproduction for *Drosophila* olfactory projection neurons (PN)