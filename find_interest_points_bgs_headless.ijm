// cd "D:\USR\Milo\Fiji_24112025" 
// fiji --headless -macro "D:\RAW_DATA\dSPIM\E4FAD2_dw\bgs_dw_reg.ijm" "D:\RAW_DATA\dSPIM\E4FAD2_dw\2025_12_02_E4FAD2_dw_Iba635_640_700_MFS05_1x1_50ms_2_out\2025_12_02_E4FAD2_dw_Iba635_640_700_MFS05_1x1_50ms_2_npy2bdv_ds124_iso0p512_MD0.xml"

xml_path = getArgument();

print("Raw argument: " + xml_path);

// Remove "file:" prefix if it exists (in case it's already there)
xml_path = replace(xml_path, "file:", "");

// Remove any accidental working directory prepending
if (indexOf(xml_path, "Fiji_24112025") >= 0) {
    // Extract the part starting with D:/RAW_DATA
    idx = indexOf(xml_path, "D:");
    if (idx >= 0) {
        xml_path = substring(xml_path, idx);
    }
}

print("Cleaned path: " + xml_path);


run("Detect Interest Points for Registration", 
    "select=" + xml_path + " " + // removed file:
    "process_angle=[All angles] " +
    "process_channel=[All channels] " +
    "process_illumination=[All illuminations] " +
    "process_tile=[All tiles] " +
    "process_timepoint=[All Timepoints] " +
    "type_of_interest_point_detection=Difference-of-Gaussian " +
    "label_interest_points=beads " +
    "set_minimal_and_maximal_intensity " +
    "limit_amount_of_detections " +
    "subpixel_localization=[3-dimensional quadratic fit] " +
    "interest_point_specification=[Advanced ...] " +
    "downsample_xy=1x " +
    "downsample_z=1x " +
    "minimal_intensity=0.0 " +
    "maximal_intensity=5000.0 " +
    "sigma=6.00000 " +
    "threshold=0.0600 " +
    "find_maxima " +
    "maximum_number=10000 " +
    "type_of_detections_to_use=Brightest " +
    "compute_on=[CPU (Java)]");




// run("Register Dataset based on Interest Points", 
//     "select=" + xml_path + " " +
//     "process_angle=[All angles] " +
//     "process_channel=[All channels] " +
//     "process_illumination=[All illuminations] " +
//     "process_tile=[All tiles] " +
//     "process_timepoint=[All Timepoints]" +
//     "registration_algorithm=[Fast descriptor-based (rotation invariant)] " +
//     "registration_in_between_views=[Only compare overlapping views (according to current transformations)] " +
//     "interest_point_inclusion=[Only compare interest points that overlap between views (according to current transformations)] " +
//     "interest_points=beads " +
//     "fix_views=[Fix first view] " +
//     "map_back_views=[Do not map back (use this if views are fixed)] " +
//     "transformation=Affine " +
//     "number_of_neighbors=3 " +
//     "redundancy=0 " +
//     "significance=10 " +
//     "allowed_error_for_ransac=5 " +
//     "inlier_factor=3 " +
//     "ransac_iterations=Normal ");



// run("Register Dataset based on Interest Points", 
//     "select=" + xml_path + " " +
//     "process_angle=[All angles] " +
//     "process_channel=[All channels] " +
//     "process_illumination=[All illuminations] " +
//     "process_tile=[All tiles] " +
//     "process_timepoint=[All Timepoints]" +
//     "registration_algorithm=[Assign closest-points with ICP (no invariance)] " +
//     "registration_in_between_views=[Only compare overlapping views (according to current transformations)] " +
//     "interest_point_inclusion=[Only compare interest points that overlap between views (according to current transformations)] " +
//     "interest_points=beads " +
//     "fix_views=[Fix first view] " +
//     "map_back_views=[Do not map back (use this if views are fixed)] " +
//     "transformation=Affine "); // going for default params by not specifying them

