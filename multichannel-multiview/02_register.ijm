// Step 2/4: register views of the reference channel based on interest points.
// Pass 1: fast descriptor-based (rotation invariant); pass 2: ICP refinement.
//
// Usage:
//   fiji --headless -macro 02_register.ijm "<path>/dataset.xml"

// ---- settings -------------------------------------------------------------
ref_channel = 0;   // must match 01_find_interest_points.ijm
// ---------------------------------------------------------------------------

xml_path = getArgument();
xml_path = replace(xml_path, "file:", "");
print("XML: " + xml_path);

common = "select=[" + xml_path + "] " +
    "process_angle=[All angles] " +
    "process_channel=[Single channel (Select from List)] " +
    "process_illumination=[All illuminations] " +
    "process_tile=[All tiles] " +
    "process_timepoint=[All Timepoints] " +
    "processing_channel=[channel " + ref_channel + "] ";

// Pass 1: descriptor-based
run("Register Dataset based on Interest Points",
    common +
    "registration_algorithm=[Fast descriptor-based (rotation invariant)] " +
    "registration_in_between_views=[Only compare overlapping views (according to current transformations)] " +
    "interest_point_inclusion=[Only compare interest points that overlap between views (according to current transformations)] " +
    "interest_points=beads " +
    "fix_views=[Fix first view] " +
    "map_back_views=[Do not map back (use this if views are fixed)] " +
    "transformation=Affine " +
    "number_of_neighbors=3 " +
    "redundancy=0 " +
    "significance=10 " +
    "allowed_error_for_ransac=5 " +
    "inlier_factor=3 " +
    "ransac_iterations=Normal");

// Pass 2: ICP refinement (default parameters)
run("Register Dataset based on Interest Points",
    common +
    "registration_algorithm=[Assign closest-points with ICP (no invariance)] " +
    "registration_in_between_views=[Only compare overlapping views (according to current transformations)] " +
    "interest_point_inclusion=[Only compare interest points that overlap between views (according to current transformations)] " +
    "interest_points=beads " +
    "fix_views=[Fix first view] " +
    "map_back_views=[Do not map back (use this if views are fixed)] " +
    "transformation=Affine");

print("Done: registration on channel " + ref_channel);
