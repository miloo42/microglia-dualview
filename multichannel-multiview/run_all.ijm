// Full pipeline: detect -> register -> duplicate transforms -> export views to OME-ZARR.
//
// Usage:
//   fiji --headless -macro run_all.ijm "<path>/dataset.xml"
//
// The XML is modified in place (registrations, bounding boxes): back it up first.

// ---- settings -------------------------------------------------------------
ref_channel = "630_700";   // channel NAME as shown in BigStitcher; registered, then copied to the others
z_crop = 1000;             // voxels to crop from each end of Z for export
temp_bbox_name = "_TempFullVolume";
cropped_bbox_name = "Cropped_Z";
// ---------------------------------------------------------------------------

xml_path = getArgument();
xml_path = replace(xml_path, "file:", "");
print("XML: " + xml_path);
if (!File.exists(xml_path))
    exit("ERROR: XML not found: " + xml_path);

select = "select=[" + xml_path + "] ";
all_views = "process_angle=[All angles] " +
            "process_illumination=[All illuminations] " +
            "process_tile=[All tiles] " +
            "process_timepoint=[All Timepoints] ";
ref_only = select + all_views +
           "process_channel=[Single channel (Select from List)] " +
           "processing_channel=[channel " + ref_channel + "] ";

// ============================================================================
// 1. Detect interest points (reference channel only)
// ============================================================================
print("\n[1/4] Detecting interest points on channel " + ref_channel);

run("Detect Interest Points for Registration",
    ref_only +
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

// ============================================================================
// 2. Register (reference channel only): descriptor-based, then ICP refinement
// ============================================================================
print("\n[2/4] Registering channel " + ref_channel);

run("Register Dataset based on Interest Points",
    ref_only +
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

run("Register Dataset based on Interest Points",
    ref_only +
    "registration_algorithm=[Assign closest-points with ICP (no invariance)] " +
    "registration_in_between_views=[Only compare overlapping views (according to current transformations)] " +
    "interest_point_inclusion=[Only compare interest points that overlap between views (according to current transformations)] " +
    "interest_points=beads " +
    "fix_views=[Fix first view] " +
    "map_back_views=[Do not map back (use this if views are fixed)] " +
    "transformation=Affine");

// ============================================================================
// 3. Copy the registration to the other channel(s)
// ============================================================================
print("\n[3/4] Duplicating transformations from channel " + ref_channel + " to all channels");

run("Duplicate Transformations",
    "apply=[One channel to other channels] " +
    select +
    "apply_to_angle=[All angles] " +
    "apply_to_illumination=[All illuminations] " +
    "apply_to_tile=[All tiles] " +
    "apply_to_timepoint=[All Timepoints] " +
    "source=" + ref_channel + " " +
    "target=[All Channels] " +
    "duplicate_which_transformations=[Replace all transformations]");

// ============================================================================
// 4. Bounding box (Z-cropped) + export each view as OME-ZARR
// ============================================================================
print("\n[4/4] Defining bounding box and exporting views");

run("Define Bounding Box",
    select + all_views +
    "process_channel=[All channels] " +
    "bounding_box=[Maximal Bounding Box spanning all transformed views] " +
    "bounding_box_name=[" + temp_bbox_name + "]");

xmlContent = File.openAsString(xml_path);

startIdx = indexOf(xmlContent, "<BoundingBoxDefinition name=\"" + temp_bbox_name + "\">");
if (startIdx == -1)
    exit("ERROR: bounding box '" + temp_bbox_name + "' not found in " + xml_path);

minStart = indexOf(xmlContent, "<min>", startIdx);
minEnd = indexOf(xmlContent, "</min>", minStart);
maxStart = indexOf(xmlContent, "<max>", startIdx);
maxEnd = indexOf(xmlContent, "</max>", maxStart);
if (minStart == -1 || minEnd == -1 || maxStart == -1 || maxEnd == -1)
    exit("ERROR: could not find <min>/<max> in bounding box definition");

minCoords = split(String.trim(substring(xmlContent, minStart + 5, minEnd)), " ");
maxCoords = split(String.trim(substring(xmlContent, maxStart + 5, maxEnd)), " ");
if (minCoords.length < 3 || maxCoords.length < 3)
    exit("ERROR: invalid bounding box coordinates in XML");

min_x = parseInt(minCoords[0]); min_y = parseInt(minCoords[1]); min_z = parseInt(minCoords[2]);
max_x = parseInt(maxCoords[0]); max_y = parseInt(maxCoords[1]); max_z = parseInt(maxCoords[2]);
print("Full box: " + min_x + "," + min_y + "," + min_z + " -> " + max_x + "," + max_y + "," + max_z);

cropped_min_z = min_z + z_crop;
cropped_max_z = max_z - z_crop;
if (cropped_min_z >= cropped_max_z)
    exit("ERROR: z_crop (" + z_crop + ") too large for Z range (" + (max_z - min_z) + ")");
print("Cropped Z range: " + cropped_min_z + " -> " + cropped_max_z);

newBboxXml = "    <BoundingBoxDefinition name=\"" + cropped_bbox_name + "\">\n" +
             "      <min>" + min_x + " " + min_y + " " + cropped_min_z + "</min>\n" +
             "      <max>" + max_x + " " + max_y + " " + cropped_max_z + "</max>\n" +
             "    </BoundingBoxDefinition>\n";

// replace a previous crop box of the same name
existing = indexOf(xmlContent, "name=\"" + cropped_bbox_name + "\"");
if (existing != -1) {
    defStart = lastIndexOf(substring(xmlContent, 0, existing), "<BoundingBoxDefinition");
    defEnd = indexOf(xmlContent, "</BoundingBoxDefinition>", existing) + 24;
    if (substring(xmlContent, defEnd, defEnd + 1) == "\n")
        defEnd = defEnd + 1;
    xmlContent = substring(xmlContent, 0, defStart) + substring(xmlContent, defEnd);
}

bboxesEnd = indexOf(xmlContent, "</BoundingBoxes>");
if (bboxesEnd == -1)
    exit("ERROR: </BoundingBoxes> not found in XML");
xmlContent = substring(xmlContent, 0, bboxesEnd) + newBboxXml + "  " + substring(xmlContent, bboxesEnd);
File.saveString(xmlContent, xml_path);

xml_dir = File.getParent(xml_path);
xml_basename = replace(File.getName(xml_path), ".xml", "");
output_dir = xml_dir + File.separator + "views";
output_zarr = output_dir + File.separator + xml_basename + "_views.zarr";
if (!File.exists(output_dir))
    File.makeDirectory(output_dir);
print("Output: " + output_zarr);

run("Image Fusion",
    select + all_views +
    "process_channel=[All channels] " +
    "bounding_box=" + cropped_bbox_name + " " +
    "downsampling=1 " +
    "interpolation=[Linear Interpolation] " +
    "fusion_type=[Avg, Blending] " +
    "pixel_type=[16-bit unsigned integer] " +
    "interest_points_for_non_rigid=[-= Disable Non-Rigid =-] " +
    "produce=[Each view] " +
    "fused_image=[OME-ZARR/N5/HDF5 export using N5-API] " +
    "define_input=[Auto-load from input data (values shown below)] " +
    "export=OME-ZARR " +
    "compression=Zstandard " +
    "zarr_dataset_path=" + output_zarr + " " +
    "show_advanced_block_size_options " +
    "block_size_x=256 " +
    "block_size_y=256 " +
    "block_size_z=256 " +
    "block_size_factor_x=1 " +
    "block_size_factor_y=1 " +
    "block_size_factor_z=1");

print("\nDone: " + output_zarr);
