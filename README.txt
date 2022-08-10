README

GENERAL INFORMATION
1. Can be used to clean the pNEUMA dataset, derive the traffic variables (speed, flow, density) using virtual loop detectors or alternatively Edie's definition. 
2. The extended traffic state, including the number of stops and the lane changes can be derived.
2. The extended TSE methods included here can be applied to the prepared dataset. These methods include a physics-informed NN, using the FD (based on the idea of the FD in traffic flow theory) that estimates the overall state (either for Edie's approach or the virtual loop detector approach) and the extended variables. 
3. The results can be processed afterwards. Some exemplary analyses are provided.

SRC
Contains the source code of the traffic state estimations (TSE) methods, along with the data preparation and evaluation.
- Analysis_0_clean_data.ipynb * for the cleansing and filtering of the data
- Analysis_1_prepare_data.ipynb * for the derivation of the traffic variables
- Analysis_2_process_data.ipynb * for the resampling and scaling of the data
- Analysis_3_process_events.ipynb * for the assignment of the extended TSE variables
- Analysis_4_TSE_Method.ipynb * for the estimation of the extended TSE using the NNFD
- Analysis_5_results.ipynb * for the results interpretation
- functions_file.py * for general functions and the sensor scenarios.
- NNet.py * for functions related to the network and physics-informed loss.

DATA (INPUT)
Contains the initial input data i.e. the pNEUMA data
- waypoints_w_dist.csv * for the trajectories (not uploaded here, can be downloaded from: https://syncandshare.lrz.de/getlink/fi6dTaQ1W6XXBoAPPsvvh9Ro/)
- veh_info_list.csv * for the vehicle information
- experiment_list_info.csv * for information about the various experiments
- polygons11.csv * for information about the 11 selected polygons (road segments, links)
- polygon_rX_Lanes * for the lane coordinates of each polygon

Note that the polygon are numbered from 0 to 11. Polygon 4 contains a higher complexity due to combining several links. Therefore it is not used within this work!

OUTPUT (all intermediate results can also be downloaded from https://syncandshare.lrz.de/getlink/fi6dTaQ1W6XXBoAPPsvvh9Ro/)
Contains the data that is returned by the code in SRC. 
- data_clean * for the cleansed data i.e. outliers removed, links filtered, etc.
- data_flow * for the traffic variables of each mode for each link
	- 'cars' * for the car mode using the mobile sensor approach (Edie's). Separate due to the penetration rate.
	- 'regular' * for all other modes and the overall state using the trajectories (Edie's)
	- 'LDs' 	* for the overall traffic state using the virtual LDs
- data_processed * for the resampled and scaled data (collapsed onto FD)
	- 'LD' * for the virtual loop detectors
	- 'MS' * for the mobile sensor approach (Edie's)
	- scalingfactors.csv * for the means and scaling factor of each link
- data_processed_events_15kmh_5sec * for the resampled and scaled data including the additional TSE parameters
- results_runs_w_events * for the extended TSE results. The errors are calculated immediately so the predictions themselves must not be saved.
