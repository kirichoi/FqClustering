# FqClustering

Python scripts for clustering neurons from morphological reconstructions using form factor F(q) curves.
The methodology has been applieed to three different morphological reconstruction datasets including [mouse primary visual cortex (V1)](https://www.nature.com/articles/s41593-019-0417-0) from the [Allen Cell Type database](https://celltypes.brain-map.org/data), [*Drosophila* olfactory projection neurons (PN)](https://www.cell.com/cell/fulltext/S0092-8674(18)30787-6) from the [TEMCA2 dataset](https://github.com/bocklab/temca2data/tree/master/geometry_analysis/data), and the [*C. elegans* nervous system](https://royalsocietypublishing.org/doi/10.1098/rstb.2017.0382) from the [Openworm c302 framework](https://github.com/openworm/c302).

- ALLEN_fq_calc.py: the Python script for calculating F(q) curves from mouse V1 reconstructions
- ALLEN_fq_cluster.py: the main Python script for the data analysis and figure reproduction for mouse V1
- ALLEN_DB.py: the Python script for querying morphological reconstructions in .swc format for neurons in the mouse V1
- ALLEN_fq: contains pre-computed F(q) curves from mouse V1 reconstructions
- dendrite_type.npy: .npy file containing dendrite type labels for the neurons in mouse V1 queried from the Allen Cell Type database
- CE_fq_calc.py: the Python script for calculating F(q) curves from the *C. elegans* nervous system
- CE_fq_cluster.py: the main Python script for the data analysis and figure reproduction for the *C. elegans* nervous system
- CE_fq_new: contains pre-computed F(q) curves from the *C. elegans* nervous system
- Dros_fq_calc.py: the Python script for calculating F(q) curves from *Drosophila* olfactory PNs
- Dros_fq_cluster.py: the main Python script for the data analysis and figure reproduction for *Drosophila* olfactory PNs
- Dros_AL_fq: contains pre-computed F(q) curves from *Drosophila* olfactory PNs at the antennal lobe
- Dros_MB_fq: contains pre-computed F(q) curves from *Drosophila* olfactory PNs at the mushroom body calyx
- Dros_LH_fq: contains pre-computed F(q) curves from *Drosophila* olfactory PNs at the lateral horn

There are few other files necessary that are not included in this repository that needs to be acquired separately.

- Morphological reconstruction files (.swc): All reconstructs are publically available at the above links. Use ALLEN_DB.py for the mouse V1.
- For the mouse V1, the supplementary file `41593_2019_417_MOESM5_ESM.xlsx`, which contains the neuron IDs that Gouwens et al. have used, is necessary for the comparison between F(q)-based and morphometry-based clustering results. This file can be found [here](https://www.nature.com/articles/s41593-019-0417-0).