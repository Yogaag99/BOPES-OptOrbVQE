# BOPES Simulation with OptOrbVQE

This script (`Bopes_OptOrbVQE.py`) demonstrates a **quantum chemistry simulation** using the **OptOrbVQE** library, focusing on calculating **Born-Oppenheimer Potential Energy Surfaces (BOPES)** for small hydrogen-based molecules.

## 📦 Requirements

This code relies on the OptOrbVQE library by Joel H. Bierman, available at:

🔗 https://github.com/JoelHBierman/OptOrbVQE

You will also need `Qiskit`, `Qiskit Nature`, `Qiskit Aer`, `PySCF`, `scipy`, and `torch`.

## 🛠 How to Set Up

### Option 1: Run Locally

1. **Clone the OptOrbVQE repository**:
   ```bash
   git clone https://github.com/JoelHBierman/OptOrbVQE.git
   ```

2. **Place the script `Bopes_OptOrbVQE.py` inside the cloned `OptOrbVQE` folder.**

3. **Install the required dependencies**, preferably inside a virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the script** using Python 3.8+:
   ```bash
   python Bopes_OptOrbVQE.py
   ```

---

### Option 2: Run in Google Colab

1. **Download and extract the OptOrbVQE library** into your Google Drive.

2. **Upload the `Bopes_OptOrbVQE.py` script** to the same location.

3. **Mount your Google Drive** in Colab and adjust the working directory:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/OptOrbVQE
   ```

4. **Run the script using**:
   ```python
   !python Bopes_OptOrbVQE.py
   ```

---

## ⚙️ Configuration

You can configure:
- The molecule type (`H2`, `H4Square`, or `H4Linear`) via `molecule_type`
- The basis set (e.g. `'sto3g'`)
- Optional orbital transformation matrix `V`:
  - Use `generate_random_partial_unitary_matrix(...)` or `generate_permutation_matrix(...)`
- Number of repetitions per geometry point (`n`)
- Output filename (automatically generated)

All configuration is done near the top of the script.

---

## 📄 What This Script Does

- Computes potential energy surfaces for hydrogen-based molecules using orbital-optimized VQE.
- Runs VQE using Qiskit components and optimizes orbital bases with `OptOrbVQE`.
- Outputs energy and timing results for varying geometries into a `.csv` file.

---

## 💡 Tips

- Use a small basis set and limited geometry range if you're testing.
- Set `V = None` to skip custom orbital projections.
- For speed, reduce the number of distance points or iterations.

---

## 📬 Credits

- Core library by [Joel H. Bierman](https://github.com/JoelHBierman) via [OptOrbVQE](https://github.com/JoelHBierman/OptOrbVQE)
- Script implementation for BOPES analysis using orbital-optimized VQE