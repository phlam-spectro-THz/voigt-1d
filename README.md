# voigt-1d
Voigt-1D window function 

Code and sample data for the following article

L. Zou and R. A. Motiyenko, Window Function for Chirped Pulse Spectrum with Enhanced Signal-to-noise Ratio and Lineshape Correction.


## Explaination

### treat_ocs_data.py
* Function: Treat the OCS FID data using different window functions, and fit the spectral lines
* Prerequisites: 
    * sample_data/OCS/OCS_freq.db 
    * sample_data/OCS/OCS_xxx.tdf         /* all available OCS data */
* Output: Generates data for Figure 3 in the main article, and Figure S1 in the Supplementary Material

### sim_snr_ab.py
* Function: Simulate theoretical SnR, FWHM and SnR/FWHM for a grid of (a, b) parameters.
* Prerequisites: None
* Output: Generates data for Figure 2 in the main article.

### sim_broad_band.py
* Function: Simulate broadband microwave and submillimeter chirped pulse spectra.
* Prerequisites: 
    * sample_data/Catalogs/CAT_nmf.cat  
    * sample_data/Catalogs/CAT_furcis_10-20.cat
* Output: Generates data for Figure 6 and Figure 7 in the main article.

### search_ocs_freq.py
* Function: Prepares a database for OCS transitions
* Prerequisites:
    * sample_data/Catalogs/OCS.cat      /* all available OCS catalogs */
* Output: sample_data/OCS/OCS_freq.db

### ch3cn_spectra.py
* Function: Treat CH3CN FID data using different window functions, and fit the spectral lines
* Prerequisites:
    * sample_data/ch3cn/ch3cn_FID_3mTorr_450ns.fit
    * sample_data/ch3cn/ch3cn_FID_3mTorr_256ns.fit
* Output: Generates data for Figure 5 in the main article, and Figure S3, S4 in the Supplementary Material

### ch3cn_fid.py
* Function: Treat the time domain CH3CN FID data
* Prerequisites:
    * sample_data/ch3cn/short_3mTorr.tdf
* Output: Generates data for Figure 4 in the main article, and Figure S2 in the Supplementary Material

### lib.py
* Function: provide library to support other scripts.
* Prerequisites: None
* Output: None
