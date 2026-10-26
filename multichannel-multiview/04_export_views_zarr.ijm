// Step 4/4: define a Z-cropped bounding box and export every view (all channels)
// as OME-ZARR, resampled into the registered coordinate system.
//
// Usage:
//   fiji --headless -macro 04_export_views_zarr.ijm "<path>/dataset.xml"
// Output: <xml_dir>/views/<xml_basename>_views.zarr

// ---- settings -------------------------------------------------------------
z_crop = 1000;                       // voxels to crop from each end of Z
temp_bbox_name = "_TempFullVolume";  // maximal box (auto-defined)
cropped_bbox_name = "Cropped_Z";     // box used for export
// ---------------------------------------------------------------------------

xml_path = getArgument();
xml_path = replace(xml_path, "file:", "");
print("XML: " + xml_path);

// --- 1. maximal bounding box over all transformed views ---
run("Define Bounding Box",
    "select=[" + xml_path + "] " +
    "process_angle=[All angles] " +
    "process_channel=[All channels] " +
    "process_illumination=[All illuminations] " +
    "process_tile=[All tiles] " +
    "process_timepoint=[All Timepoints] " +
    "bounding_box=[Maximal Bounding Box spanning all transformed views] " +
    "bounding_box_name=[" + temp_bbox_name + "]");

// --- 2. read its coordinates back from the XML ---
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

// --- 3. add the Z-cropped box to the XML ---
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

// --- output path ---
xml_dir = File.getParent(xml_path);
xml_basename = replace(File.getName(xml_path), ".xml", "");
output_dir = xml_dir + File.separator + "views";
output_zarr = output_dir + File.separator + xml_basename + "_views.zarr";
if (!File.exists(output_dir))
    File.makeDirectory(output_dir);
print("Output: " + output_zarr);

// --- 4. export each view ---
run("Image Fusion",
    "select=[" + xml_path + "] " +
    "process_angle=[All angles] " +
    "process_channel=[All channels] " +
    "process_illumination=[All illuminations] " +
    "process_tile=[All tiles] " +
    "process_timepoint=[All Timepoints] " +
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

print("Done: " + output_zarr);
