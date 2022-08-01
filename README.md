# TRB_Extended_TSE
Code for the TRB-paper: EXTENDED URBAN TRAFFIC STATE ESTIMATION USING DIFFERENT SENSOR STRATEGIES
README

GENERAL INFORMATION
1. Can be used to clean the pNEUMA dataset, derived the traffic variables (speed, flow, density) using virtual loop detectors or alternatively Edie's definition. 
2. The TSE methods included here can be applied to the prepared dataset. These methods include NN which both estimate the flow/density and the extended parameters (number of stops, number of lane changes) of all modes. A physics-informed NNFD (based on the idea of the FD in traffic flow theory) that estimates the overall state (either for Edie's approach or the virtual loop detector approach). 
3. The results can be processed afterwards. Some exemplary analyses are provided.

SRC
Contains the source code of the traffic state estimations (TSE) methods, along with the data preparation and evaluation.
- Analysis_0_clean_data.ipynb * for the cleansing and filtering of the data
- Analysis_1_prepare_data.ipynb * for the derivation of the traffic variables
- Analysis_2_process_data.ipynb * for the resampling and scaling of the data
- Analysis_3_MLR.ipynb * for the MLR
- Analysis_4_NN.ipynb * for the NN
- Analysis_5_MLRFD.ipynb * for the MLRFD - physics-informed MLR
- Analysis_6_NNFD.ipynb * for the NNFD - physics-informed NN
- Analysis_7_Results.ipynb * for analysis of the results
- functions_file.py * for general functions and the sensor scenarios, mainly for Analysis 0-2.
- NNet.py * for functions related to the networks and physics-informed loss, mainly for Analysis 3-6.

DATA (INPUT)
Contains the initial input data i.e. the pNEUMA data
- waypoints_w_dist.csv * for the trajectories (not in drive)
- veh_info_list.csv * for the vehicle information
- experiment_list_info.csv * for information about the various experiments
- polygons11.csv * for information about the 11 selected polygons (road segments, links)

OUTPUT
Contains the data the is returned by the code in SRC. 
- data_clean * for the cleansed data i.e. outliers removed, links filtered, etc.
- data_flow * for the traffic variables of each mode for each link
	- 'cars' * for the car mode using the mobile sensor approach (Edie's). Separate due to the penetration rate.
	- 'regular' * for all other modes and the overall state using the trajectories (Edie's)
	- 'LDs' 	* for the overall traffic state using the virtual LDs
- data_processed * for the resampled and scaled data (collapsed onto FD)
	- 'LD' * for the virtual loop detectors
	- 'MS' * for the mobile sensor approach (Edie's)
	- scalingfactors.csv * for the means and scaling factor of each link
- results_runs * for the TSE results. The errors are calculated immediately so the predictions themselves must not be saved.

