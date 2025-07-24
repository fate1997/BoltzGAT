# Processed databases
Please download the processed databases from [Google Drive](https://drive.google.com/drive/folders/1jSPmP2uwgJyAga5AHrerDuKdtsS5UL8v?usp=sharing) to `database/processed`.

# Summary of the datasets
In this work, we process the databases from Yaws's Handbook[1], and finally obtained 18 datasets in total (the details are shown in the Table below). Each dataset is stored in a pickle file. Each pickle file contains a list of dictionaries. Each dictionary contains the following keys:
- `No.`: The index of the molecule in the dataset.
- `Name`: The name of the molecule.
- `SMILES`: The SMILES representation of the molecule.
- `Temperature`: A list represents the temperatures at which the properties are measured.
- `y`: A list represents the property values at the corresponding temperatures.

|File| Unit|Description|Number of molecules|Number of data points|
|:---:|:---:|---|:---:|:---:|
|viscosity_L|	mPa·s	|Viscosity of liquids|	9830|	37152|
|viscosity_G|	mPa·s	|Viscosity of gases|	8347|	66776|
|thermal_cond_L|	W/(m·K)|	Thermal conductivity of liquids|	8443	|67544|
|thermal_cond_G|	W/(m·K)|	Thermal conductivity of gases|	8349|	66792|
|diffusion_coef_water|	cm^2/s|	Diffusion coefficient in water	|6566	|52528|
|diffusion_coef_air|	cm^2/s|	Diffusion coefficient in air|	6554|	52432|
|Cp_L	|J/(mol·K)|	Specific heat capacity of liquids|	4716	|37728|
|Cp_G|	J/(mol·K)|	Specific heat capacity of gases	|4699|37592|
|Cp_S|	J/(mol·K)	|Specific heat capacity of solids	|4041|	32328|
|surface_tension	|dyn/cm|	Surface tension|	4722|	37776|
|density	|g/cm3	|Density of liquids	|4369	|34952|
|gas entropy	|J/(mol·K)|	Entropy of ideal gas|	4496	|35968|
|delta_Hf|	kJ/mol|	Enthalpy formation of ideal gas|	4082|	32656|
|delta_Gf|	kJ/mol	|Gibbs energy of formation of ideal gas|	3790	|30320|
|delta_Af|	kJ/mol	|Helmholtz energy of formation of ideal gas	|4152	|33216|
|delta_Uf	|kJ/mol|	Internal energy of formation of ideal gas	|4079	|32632|
|delta_Sf	|J/(mol·K)|	Entropy formation of gas|	4498|	35984|
|H_vap|	kJ/mol	|Enthalpy of vaporization	|4694	|37552|

[1] Carl L. Yaws, Yaws' Handbook of Thermodynamic and Physical Properties of Chemical Compounds. Knovel, 2003.
