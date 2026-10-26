// Step 1/3: detect interest points (beads) on all views.
//
// Usage:
//   fiji --headless -macro 01_find_interest_points.ijm "<path>/dataset.xml"

xml_path = getArgument();
xml_path = replace(xml_path, "file:", "");
// headless Fiji can prepend its own directory to the argument: keep the last absolute Windows path
xml_path = replace(xml_path, "^.+([A-Za-z]:[\\\\/])", "$1");
print("XML: " + xml_path);
if (!File.exists(xml_path))
    exit("ERROR: XML not found: " + xml_path);

run("Detect Interest Points for Registration",
    "select=[" + xml_path + "] " +
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

print("Done: interest point detection");
