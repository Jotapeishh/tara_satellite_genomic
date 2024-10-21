## Satellite Features Extracted from MODIS-NASA

This section provides detailed explanations of the satellite features extracted from MODIS-NASA data. The features are categorized for clarity.

### Table of Contents

- [Chlorophyll](#chlorophyll)
- [Fluorescence Line Height (FLH)](#fluorescence-line-height-flh)
- [Inherent Optical Properties (IOP)](#inherent-optical-properties-iop)
  - [Absorption Coefficients (`IOP.a_{wavelength}`)](#absorption-coefficients-iopa_wavelength)
  - [Backscattering Coefficients (`IOP.bb_{wavelength}`)](#backscattering-coefficients-iopbb_wavelength)
  - [Dissolved and Detrital Absorption (`IOP.adg_*`)](#dissolved-and-detrital-absorption-iopadg_)
  - [Phytoplankton Absorption (`IOP.aph_*`)](#phytoplankton-absorption-iopaph_)
  - [Particle Backscattering (`IOP.bbp_*`)](#particle-backscattering-iopbbp_)
- [Diffuse Attenuation Coefficient](#diffuse-attenuation-coefficient)
- [Sea Surface Temperature (SST)](#sea-surface-temperature-sst)
- [Photosynthetically Available Radiation (PAR)](#photosynthetically-available-radiation-par)
- [Particulate Inorganic Carbon (PIC)](#particulate-inorganic-carbon-pic)
- [Particulate Organic Carbon (POC)](#particulate-organic-carbon-poc)
- [Remote Sensing Reflectance (RRS)](#remote-sensing-reflectance-rrs)
- [Aerosol Optical Thickness and Angstrom Exponent](#aerosol-optical-thickness-and-angstrom-exponent)

---

### Chlorophyll

- **`CHL.chlor_a`**

  The concentration of chlorophyll-a in the water column, measured in milligrams per cubic meter (mg/m³). Chlorophyll-a is a pigment found in phytoplankton and is used as an indicator of biomass and primary productivity in marine ecosystems.

### Fluorescence Line Height (FLH)

- **`FLH.nflh`**

  Normalized Fluorescence Line Height, which quantifies the fluorescence emitted by chlorophyll-a when excited by sunlight. It's used to estimate chlorophyll concentration and assess phytoplankton health.

- **`FLH.ipar`**

  Incident Photosynthetically Available Radiation, representing the amount of sunlight (in the photosynthetically active wavelengths) reaching the ocean surface, measured in Einsteins per square meter per day (Ein/m²/day).

### Inherent Optical Properties (IOP)

Inherent Optical Properties are properties of water that depend solely on the medium and its constituents, independent of the ambient light field.

#### Absorption Coefficients (`IOP.a_{wavelength}`)

Total absorption coefficients at specific wavelengths, measured in inverse meters (m⁻¹). They represent how much light is absorbed per unit distance at each wavelength.

- **Wavelengths:**
  - `IOP.a_412`
  - `IOP.a_443`
  - `IOP.a_469`
  - `IOP.a_488`
  - `IOP.a_531`
  - `IOP.a_547`
  - `IOP.a_555`
  - `IOP.a_645`
  - `IOP.a_667`
  - `IOP.a_678`

#### Backscattering Coefficients (`IOP.bb_{wavelength}`)

Total backscattering coefficients at specific wavelengths, measured in inverse meters (m⁻¹). They represent the fraction of light scattered back towards the direction it came from per unit distance.

- **Wavelengths:**
  - `IOP.bb_412`
  - `IOP.bb_443`
  - `IOP.bb_469`
  - `IOP.bb_488`
  - `IOP.bb_531`
  - `IOP.bb_547`
  - `IOP.bb_555`
  - `IOP.bb_645`
  - `IOP.bb_667`
  - `IOP.bb_678`

#### Dissolved and Detrital Absorption (`IOP.adg_*`)

- **`IOP.adg_443`**

  Absorption coefficient for combined dissolved and detrital matter at 443 nm.

- **`IOP.adg_unc_443`**

  Uncertainty in the absorption coefficient for dissolved and detrital matter at 443 nm.

- **`IOP.adg_s`**

  Spectral slope of the absorption coefficient for dissolved and detrital matter, indicating how absorption changes with wavelength.

#### Phytoplankton Absorption (`IOP.aph_*`)

- **`IOP.aph_443`**

  Absorption coefficient for phytoplankton at 443 nm.

- **`IOP.aph_unc_443`**

  Uncertainty in the absorption coefficient for phytoplankton at 443 nm.

#### Particle Backscattering (`IOP.bbp_*`)

- **`IOP.bbp_443`**

  Particle backscattering coefficient at 443 nm, representing scattering by particles in the water.

- **`IOP.bbp_unc_443`**

  Uncertainty in the particle backscattering coefficient at 443 nm.

- **`IOP.bbp_s`**

  Spectral slope of the particle backscattering coefficient, indicating how backscattering changes with wavelength.

### Diffuse Attenuation Coefficient

- **`KD.Kd_490`**

  Diffuse attenuation coefficient at 490 nm, measured in inverse meters (m⁻¹). It quantifies the rate at which light diminishes with depth due to absorption and scattering.

### Sea Surface Temperature (SST)

- **`NSST.sst`**
- **`SST.sst`**

  Sea Surface Temperature, measured in degrees Celsius (°C). It represents the temperature of the ocean's surface layer.

### Photosynthetically Available Radiation (PAR)

- **`PAR.par`**

  Photosynthetically Available Radiation, representing the quantum energy available for photosynthesis, measured in Einsteins per square meter per day (Ein/m²/day).

### Particulate Inorganic Carbon (PIC)

- **`PIC.pic`**

  Concentration of particulate inorganic carbon, primarily from calcium carbonate particles like coccolithophores, measured in moles per cubic meter (mol/m³).

### Particulate Organic Carbon (POC)

- **`POC.poc`**

  Concentration of particulate organic carbon in the water column, measured in milligrams per cubic meter (mg/m³). It indicates the amount of organic matter, including phytoplankton and detritus.

### Remote Sensing Reflectance (RRS)

Remote Sensing Reflectance is the ratio of the water-leaving radiance to the downwelling irradiance just above the water surface, measured in inverse steradians (sr⁻¹). It is used to derive various ocean color products.

- **Wavelengths:**
  - `RRS.Rrs_412`
  - `RRS.Rrs_443`
  - `RRS.Rrs_469`
  - `RRS.Rrs_488`
  - `RRS.Rrs_531`
  - `RRS.Rrs_547`
  - `RRS.Rrs_555`
  - `RRS.Rrs_645`
  - `RRS.Rrs_667`
  - `RRS.Rrs_678`

### Aerosol Optical Thickness and Angstrom Exponent

- **`RRS.aot_869`**

  Aerosol Optical Thickness at 869 nm, representing the degree to which aerosols prevent the transmission of light by absorption or scattering.

- **`RRS.angstrom`**

  Angstrom Exponent, a parameter indicating the wavelength dependency of aerosol optical thickness, used to infer particle size distribution.

---

For further details on the algorithms and data processing methods, refer to the [MODIS Ocean Color Documentation](https://oceancolor.gsfc.nasa.gov/docs/).
