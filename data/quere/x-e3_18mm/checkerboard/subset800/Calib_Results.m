% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 615.400774324724921 ; 613.002426110905844 ];

%-- Principal point:
cc = [ 397.448855891933306 ; 260.956455397324476 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ -0.051102159247368 ; 0.059105713452886 ; -0.003543345540656 ; -0.001539874018630 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 0.595374910300209 ; 0.560396694808595 ];

%-- Principal point uncertainty:
cc_error = [ 0.649465886862714 ; 0.530633441426426 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.002410772725998 ; 0.006913746992412 ; 0.000264147532797 ; 0.000356187252736 ; 0.000000000000000 ];

%-- Image size:
nx = 800;
ny = 531;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 20;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ 2.195064e+00 ; 2.208541e+00 ; -3.634669e-02 ];
Tc_1  = [ -1.716349e+03 ; -1.136458e+03 ; 3.697705e+03 ];
omc_error_1 = [ 7.498987e-04 ; 7.801266e-04 ; 1.620962e-03 ];
Tc_error_1  = [ 4.008645e+00 ; 3.286076e+00 ; 3.934512e+00 ];

%-- Image #2:
omc_2 = [ 2.026202e+00 ; 2.011602e+00 ; 1.426713e-01 ];
Tc_2  = [ -1.158695e+03 ; -1.196118e+03 ; 3.929182e+03 ];
omc_error_2 = [ 8.622831e-04 ; 8.205101e-04 ; 1.568116e-03 ];
Tc_error_2  = [ 4.242565e+00 ; 3.408977e+00 ; 4.085217e+00 ];

%-- Image #3:
omc_3 = [ 1.778300e+00 ; 1.800226e+00 ; 5.346391e-01 ];
Tc_3  = [ -7.207388e+02 ; -1.303956e+03 ; 3.720507e+03 ];
omc_error_3 = [ 9.585753e-04 ; 8.167586e-04 ; 1.277510e-03 ];
Tc_error_3  = [ 4.022104e+00 ; 3.214926e+00 ; 4.147296e+00 ];

%-- Image #4:
omc_4 = [ -2.061942e+00 ; -2.140337e+00 ; 5.189808e-01 ];
Tc_4  = [ -1.871644e+03 ; -1.400714e+03 ; 4.753701e+03 ];
omc_error_4 = [ 9.516392e-04 ; 7.107034e-04 ; 1.553098e-03 ];
Tc_error_4  = [ 5.105657e+00 ; 4.137320e+00 ; 4.545422e+00 ];

%-- Image #5:
omc_5 = [ -1.912056e+00 ; -2.012631e+00 ; 7.001655e-01 ];
Tc_5  = [ -2.433307e+03 ; -1.466302e+03 ; 6.233213e+03 ];
omc_error_5 = [ 1.066593e-03 ; 7.843516e-04 ; 1.518698e-03 ];
Tc_error_5  = [ 6.778694e+00 ; 5.555854e+00 ; 5.611344e+00 ];

%-- Image #6:
omc_6 = [ -1.899126e+00 ; -1.903865e+00 ; 7.947729e-01 ];
Tc_6  = [ -3.095442e+03 ; -1.132177e+03 ; 6.657550e+03 ];
omc_error_6 = [ 1.098444e-03 ; 8.123042e-04 ; 1.426079e-03 ];
Tc_error_6  = [ 7.337788e+00 ; 6.045726e+00 ; 6.024474e+00 ];

%-- Image #7:
omc_7 = [ 1.862672e+00 ; 1.982731e+00 ; -8.921257e-01 ];
Tc_7  = [ -3.050126e+03 ; -1.503927e+03 ; 7.746908e+03 ];
omc_error_7 = [ 6.514963e-04 ; 1.190052e-03 ; 1.618022e-03 ];
Tc_error_7  = [ 8.370431e+00 ; 6.973261e+00 ; 7.352902e+00 ];

%-- Image #8:
omc_8 = [ 1.834888e+00 ; 1.791030e+00 ; -7.016818e-01 ];
Tc_8  = [ -1.505683e+03 ; -1.252856e+03 ; 7.027041e+03 ];
omc_error_8 = [ 7.464613e-04 ; 1.047096e-03 ; 1.497461e-03 ];
Tc_error_8  = [ 7.487372e+00 ; 6.081043e+00 ; 6.447473e+00 ];

%-- Image #9:
omc_9 = [ 1.922615e+00 ; 1.530186e+00 ; -4.751988e-01 ];
Tc_9  = [ 2.177619e+00 ; -1.187345e+03 ; 7.129728e+03 ];
omc_error_9 = [ 9.018884e-04 ; 9.339444e-04 ; 1.478198e-03 ];
Tc_error_9  = [ 7.639238e+00 ; 6.104807e+00 ; 6.813616e+00 ];

%-- Image #10:
omc_10 = [ 1.741408e+00 ; 1.603245e+00 ; -7.756848e-01 ];
Tc_10  = [ -1.149392e+03 ; -9.297227e+02 ; 9.182655e+03 ];
omc_error_10 = [ 8.711089e-04 ; 1.112411e-03 ; 1.485847e-03 ];
Tc_error_10  = [ 9.737720e+00 ; 7.945380e+00 ; 8.659446e+00 ];

%-- Image #11:
omc_11 = [ 1.914152e+00 ; 1.722397e+00 ; -4.301851e-01 ];
Tc_11  = [ -8.447127e+02 ; -7.914196e+02 ; 5.736759e+03 ];
omc_error_11 = [ 8.146707e-04 ; 8.824491e-04 ; 1.489557e-03 ];
Tc_error_11  = [ 6.083243e+00 ; 4.888693e+00 ; 5.350066e+00 ];

%-- Image #12:
omc_12 = [ 1.957347e+00 ; 1.983097e+00 ; -2.250309e-01 ];
Tc_12  = [ -7.580678e+02 ; -1.062913e+03 ; 5.600365e+03 ];
omc_error_12 = [ 8.785919e-04 ; 9.338306e-04 ; 1.763718e-03 ];
Tc_error_12  = [ 5.948961e+00 ; 4.778436e+00 ; 5.584115e+00 ];

%-- Image #13:
omc_13 = [ 1.985769e+00 ; 2.051764e+00 ; 1.050878e+00 ];
Tc_13  = [ -6.537718e+02 ; -5.520802e+02 ; 3.459287e+03 ];
omc_error_13 = [ 1.210116e-03 ; 6.357007e-04 ; 1.483606e-03 ];
Tc_error_13  = [ 3.700714e+00 ; 3.026936e+00 ; 4.374069e+00 ];

%-- Image #14:
omc_14 = [ 1.923642e+00 ; 1.989988e+00 ; 9.506025e-01 ];
Tc_14  = [ -7.196084e+01 ; -7.909456e+02 ; 4.006450e+03 ];
omc_error_14 = [ 1.192744e-03 ; 7.293401e-04 ; 1.439598e-03 ];
Tc_error_14  = [ 4.294175e+00 ; 3.471897e+00 ; 4.784797e+00 ];

%-- Image #15:
omc_15 = [ -1.468998e+00 ; -2.113624e+00 ; 6.819228e-01 ];
Tc_15  = [ -1.466006e+03 ; -1.790242e+03 ; 6.119621e+03 ];
omc_error_15 = [ 9.063332e-04 ; 9.449277e-04 ; 1.313077e-03 ];
Tc_error_15  = [ 6.577663e+00 ; 5.348567e+00 ; 5.235699e+00 ];

%-- Image #16:
omc_16 = [ 1.355460e+00 ; 2.578206e+00 ; -1.061889e+00 ];
Tc_16  = [ -6.641904e+02 ; -2.107552e+03 ; 7.188825e+03 ];
omc_error_16 = [ 6.044582e-04 ; 1.207741e-03 ; 1.582440e-03 ];
Tc_error_16  = [ 7.739789e+00 ; 6.248105e+00 ; 5.989980e+00 ];

%-- Image #17:
omc_17 = [ -2.333389e-01 ; -2.874138e+00 ; 1.088788e+00 ];
Tc_17  = [ 1.495269e+03 ; -1.849220e+03 ; 7.113031e+03 ];
omc_error_17 = [ 7.708742e-04 ; 1.172989e-03 ; 1.462458e-03 ];
Tc_error_17  = [ 7.595118e+00 ; 6.206686e+00 ; 6.181921e+00 ];

%-- Image #18:
omc_18 = [ 1.330164e+00 ; 2.642312e+00 ; -6.679713e-01 ];
Tc_18  = [ -1.006490e+03 ; -2.196879e+03 ; 6.326844e+03 ];
omc_error_18 = [ 5.248626e-04 ; 1.247256e-03 ; 1.637051e-03 ];
Tc_error_18  = [ 6.814430e+00 ; 5.467430e+00 ; 5.716347e+00 ];

%-- Image #19:
omc_19 = [ -2.136548e+00 ; -2.220788e+00 ; -2.066139e-01 ];
Tc_19  = [ -3.999981e+03 ; -1.076400e+03 ; 9.070297e+03 ];
omc_error_19 = [ 2.063525e-03 ; 1.842475e-03 ; 4.317721e-03 ];
Tc_error_19  = [ 9.717417e+00 ; 8.217508e+00 ; 1.070439e+01 ];

%-- Image #20:
omc_20 = [ -2.174003e+00 ; -2.167827e+00 ; -1.990247e-01 ];
Tc_20  = [ 1.035242e+03 ; -1.255788e+03 ; 8.537836e+03 ];
omc_error_20 = [ 1.317515e-03 ; 1.731714e-03 ; 3.064832e-03 ];
Tc_error_20  = [ 9.286851e+00 ; 7.460023e+00 ; 1.014230e+01 ];

