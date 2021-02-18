# voigt-1d
Voigt-1D window function 

Code and sample data for the following article

L. Zou and R. A. Motiyenko, "Window Function for Chirped Pulse Spectrum with Enhanced Signal-to-noise Ratio and Lineshape Correction"


## Explaination

### Jupyter notebook
`test_numerical_methods.ipynb` includes codes for basic tests of the numerical methods
used in this study. 

### Script execution order:
`||` means independent and can be run in parallel, `>>` means sequential execution.
Each list item can be run in parallel
* `kaiser_theory.py || v1d_theory.py || v1d_avg_vs_optimal.py`
* `search_ocs_freq.py >> treat_ocs_data.py`
* `ch3cn_fid.py >> ch3cn_spectra.py`
* `sim_broad_band.py`

### kaiser_theory.py, v1d_theory.py
* Function: Simulate theoretical SnR, FWHM and SnR/FWHM for a given FID with 
(a0, b0) envelope parameter 
  * parameter space (a, b) for Voigt-1D window
  * parameter space (tlen, pi*alpha) for Kaiser window
* Prerequisites: None
* Output: 
    * sample_data/v1d_theory_xx-xx.dat
    * sample_data/kaiser_theory_xx-xx.dat
* Correspond to:
    * Figure 3 in the main article.
    
 ### v1d_avg_vs_optimal.py
 * Function. Simulate theoretical SnR, FWHM for a broadband FID (a0, b0) grid
 with "averaged" Voigt-1D window parameter and optimal Voigt-1D window parameter
 * Prerequisites: None
 * Output:
    * sample_data/SI_v1d_fixed_ab_xxx.dat
 * Correspond to:
    * Figure S1 in the Supplementary Material
        
### search_ocs_freq.py
* Function: Prepares a database for OCS transitions
* Prerequisites:                        /* all available OCS catalogs */
    * sample_data/Catalogs/OCS_v=0.cat      
    * sample_data/Catalogs/OCS_v2=1.cat      
    * sample_data/Catalogs/18OCS.cat    
    * sample_data/Catalogs/O13CS.cat      
    * sample_data/Catalogs/OC33S.cat      
    * sample_data/Catalogs/OC34S.cat      
* Output: 
    * sample_data/OCS/OCS_freq.db
    
### treat_ocs_data.py
* Function: Treat the OCS FID data using different window functions, and fit the spectral lines
* Prerequisites: 
    * sample_data/OCS/OCS_freq.db 
    * sample_data/OCS/OCS_xxx.tdf         /* all available OCS data */
* Output: 
    * sample_data/OCS_voigt1d_vs_kaiser_0zp.log
    * sample_data/OCS_voigt1d_vs_kaiser_20kzp.log
    * sample_data/OCS/OCS_xxx.fit
* Correspond to:
    * Figure 4 in the main article
    * Figure S2 in the Supplementary Material

### ch3cn_fid.py
* Function: Treat the time domain CH3CN FID data
* Prerequisites:
    * sample_data/ch3cn/short_3mTorr.tdf
* Output: 
    * sample_data/ch3cn/ch3cn_FID_3mTorr_450ns.fit
    * sample_data/ch3cn/ch3cn_FID_3mTorr_256ns.fit
* Correspond to: 
    * Figure 5 in the main article
    * Figure S3 in the Supplementary Material

### ch3cn_spectra.py
* Function: Treat CH3CN FID data using different window functions, and fit the spectral lines
* Prerequisites:
    * sample_data/ch3cn/ch3cn_FID_3mTorr_450ns.fit
    * sample_data/ch3cn/ch3cn_FID_3mTorr_256ns.fit
* Output: 
* Correspond to: 
    * Figure 6 in the main article
    * Figure S4, S5 in the Supplementary Material

### sim_broad_band.py
* Function: Simulate broadband microwave and submillimeter chirped pulse spectra.
* Prerequisites: 
    * sample_data/Catalogs/CAT_nmf.cat  
    * sample_data/Catalogs/CAT_furcis_2-20.cat
* Output: 
    * sample_data/broadband_sim_chirp_cis-furfural_2-10_FID.dat
    * sample_data/broadband_sim_chirp_cis-furfural_2-10_FT.txt
    * sample_data/broadband_sim_chirp_N-MF_640-650_FID.dat
    * sample_data/broadband_sim_chirp_N-MF_640-650_FT.txt
* Correspond to:
    * Figure 7, 8 in the main article. 
* Note:
    * Data are slightly different in each run because of the randomly generated
    noise.  

### lib.py
* Function: provide functions to support other scripts.
* Prerequisites: None
* Output: None
