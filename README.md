# alibz
Laser induced breakdown spectroscopy data analysis

## Installation

Use `pip` to install the required libraries. The repository includes a
`dependencies.txt` file with the core packages:

```bash
pip install -r dependencies.txt
```

## Usage

The `peaky_demo_v1.ipynb` notebook demonstrates the typical workflow for
finding and indexing spectral peaks.

1. **Open the demo notebook**

   Launch Jupyter and open `peaky_demo_v1.ipynb`.

2. **Import the analysis tools**

   ```python
   from peaky_finder import PeakyFinder
   from peaky_indexer import PeakyIndexer
   ```

3. **Create the Finder and Indexer**

   Provide the path to your raw spectrum data and initialize the classes:

   ```python
   data_dir = "path/to/raw/data"
   finder = PeakyFinder(data_dir)
   indexer = PeakyIndexer(finder)
   ```

4. **Load data and fit spectra**

   ```python
   finder.data.load_data()
   results = finder.fit_spectrum_data(sample_index, n_sigma=1)
   ```

5. **Match peaks to database entries**

   ```python
   matches = indexer.peak_match(
       results["sorted_parameter_array"],
       element_list=["Li", "Na", "Ca", "K", "Rb", "Cs", "Ba"],
   )
   ```

6. **Visualize matched peaks (optional)**

   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   li_peaks = np.array(list(matches["Li"].ions[1.0].values()))
   plt.scatter(li_peaks[:, 0], li_peaks[:, 1])
   ```

These steps mirror the workflow in `peaky_demo_v1.ipynb` and provide a
starting point for analyzing laser-induced breakdown spectroscopy data.
