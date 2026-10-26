// cd "D:\USR\Milo\Fiji_24112025" 
// fiji --headless -macro "D:\RAW_DATA\dSPIM\E4FAD2_dw\save_views_tozarr_dw_bgs_headless.ijm" "D:\RAW_DATA\dSPIM\E4FAD2_dw\2025_12_02_E4FAD2_dw_Iba635_640_700_MFS05_1x1_50ms_2_out\2025_12_02_E4FAD2_dw_Iba635_640_700_MFS05_1x1_50ms_2_npy2bdv_ds124_iso0p512_MD0.xml"

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


z_crop = 1000;  // voxels to crop from each end of Z

// Bounding box names (used internally)
temp_bbox_name = "_TempFullVolume";
cropped_bbox_name = "Cropped_Z";


// ============================================
// STEP 1: Define maximal bounding box
// ============================================
print("Step 1: Defining maximal bounding box...");

run("Define Bounding Box", 
    "select=" + xml_path + " " +
    "process_angle=[All angles] " +
    "process_channel=[All channels] " +
    "process_illumination=[All illuminations] " +
    "process_tile=[All tiles] " +
    "process_timepoint=[All Timepoints] " +
    "bounding_box=[Maximal Bounding Box spanning all transformed views] " +
    "bounding_box_name=[" + temp_bbox_name + "]");

print("Maximal bounding box defined and saved to XML.");

// ============================================
// STEP 2: Parse the XML to get bounding box coordinates
// ============================================
print("Step 2: Parsing XML to extract bounding box coordinates...");

xmlContent = File.openAsString(xml_path);

// Find the bounding box we just created
// XML format is:
// <BoundingBoxDefinition name="_TempFullVolume">
//   <min>x y z</min>
//   <max>x y z</max>
// </BoundingBoxDefinition>

// Find the bounding box definition by name
searchStr = "<BoundingBoxDefinition name=\"" + temp_bbox_name + "\">";
startIdx = indexOf(xmlContent, searchStr);

if (startIdx == -1) {
    // Try alternative: might be using existing bounding box
    // Let's also check for any BoundingBoxDefinition
    searchStr = "<BoundingBoxDefinition name=\"";
    startIdx = indexOf(xmlContent, searchStr);
    if (startIdx == -1) {
        exit("ERROR: Could not find any bounding box in XML file: " + xml_path);
    }
    print("WARNING: Could not find '" + temp_bbox_name + "', using first available bounding box.");
}

// Find min coordinates after the bounding box definition start
minStart = indexOf(xmlContent, "<min>", startIdx);
minEnd = indexOf(xmlContent, "</min>", minStart);
if (minStart == -1 || minEnd == -1) {
    exit("ERROR: Could not find <min> tags in bounding box definition");
}
minStr = substring(xmlContent, minStart + 5, minEnd);
minStr = String.trim(minStr);
minCoords = split(minStr, " ");

if (minCoords.length < 3) {
    exit("ERROR: Invalid min coordinates format: " + minStr);
}

min_x = parseInt(minCoords[0]);
min_y = parseInt(minCoords[1]);
min_z = parseInt(minCoords[2]);

// Find max coordinates
maxStart = indexOf(xmlContent, "<max>", startIdx);
maxEnd = indexOf(xmlContent, "</max>", maxStart);
if (maxStart == -1 || maxEnd == -1) {
    exit("ERROR: Could not find <max> tags in bounding box definition");
}
maxStr = substring(xmlContent, maxStart + 5, maxEnd);
maxStr = String.trim(maxStr);
maxCoords = split(maxStr, " ");

if (maxCoords.length < 3) {
    exit("ERROR: Invalid max coordinates format: " + maxStr);
}

max_x = parseInt(maxCoords[0]);
max_y = parseInt(maxCoords[1]);
max_z = parseInt(maxCoords[2]);

print("Original bounding box:");
print("  Min: X=" + min_x + ", Y=" + min_y + ", Z=" + min_z);
print("  Max: X=" + max_x + ", Y=" + max_y + ", Z=" + max_z);
print("  Dimensions: " + (max_x - min_x) + " x " + (max_y - min_y) + " x " + (max_z - min_z) + " voxels");

// ============================================
// STEP 3: Calculate and define cropped bounding box
// ============================================
print("Step 3: Defining cropped bounding box...");

// Calculate cropped Z values
cropped_min_z = min_z + z_crop;
cropped_max_z = max_z - z_crop;

// Validate that cropping doesn't result in invalid bounds
if (cropped_min_z >= cropped_max_z) {
    exit("ERROR: Z crop value (" + z_crop + ") is too large for dataset Z range (" + (max_z - min_z) + ")");
}

print("Cropped bounding box:");
print("  Min: X=" + min_x + ", Y=" + min_y + ", Z=" + cropped_min_z);
print("  Max: X=" + max_x + ", Y=" + max_y + ", Z=" + cropped_max_z);
print("  New Z range: " + (cropped_max_z - cropped_min_z) + " voxels (cropped " + z_crop + " from each end)");

// ============================================
// STEP 3b: Add cropped bounding box to XML
// ============================================
print("");
print("Step 3b: Adding cropped bounding box to XML...");

// Create the new bounding box XML element
newBboxXml = "    <BoundingBoxDefinition name=\"" + cropped_bbox_name + "\">\n";
newBboxXml += "      <min>" + min_x + " " + min_y + " " + cropped_min_z + "</min>\n";
newBboxXml += "      <max>" + max_x + " " + max_y + " " + cropped_max_z + "</max>\n";
newBboxXml += "    </BoundingBoxDefinition>\n";

// Check if this bounding box already exists, if so remove it first
existingCroppedStart = indexOf(xmlContent, "name=\"" + cropped_bbox_name + "\"");
if (existingCroppedStart != -1) {
    print("  Removing existing '" + cropped_bbox_name + "' bounding box...");
    // Find the start of this BoundingBoxDefinition
    defStart = lastIndexOf(substring(xmlContent, 0, existingCroppedStart), "<BoundingBoxDefinition");
    defEnd = indexOf(xmlContent, "</BoundingBoxDefinition>", existingCroppedStart) + 24;
    // Also include trailing newline if present
    if (substring(xmlContent, defEnd, defEnd + 1) == "\n") {
        defEnd = defEnd + 1;
    }
    xmlContent = substring(xmlContent, 0, defStart) + substring(xmlContent, defEnd);
}

// Find the closing </BoundingBoxes> tag and insert before it
bboxesEnd = indexOf(xmlContent, "</BoundingBoxes>");
if (bboxesEnd == -1) {
    exit("ERROR: Could not find </BoundingBoxes> tag in XML");
}

// Insert the new bounding box
xmlContent = substring(xmlContent, 0, bboxesEnd) + newBboxXml + "  " + substring(xmlContent, bboxesEnd);

// Write the modified XML back to file
File.saveString(xmlContent, xml_path);
print("  Cropped bounding box added to XML file.");




// ============================================
// PARSE XML PATH TO CREATE OUTPUT PATH
// ============================================
// Input:  D:\RAW_DATA\dSPIM\E4FAD2_dw\2025_12_02_...\2025_12_02_..._npy2bdv_ds124_iso0p512_MD0.xml
// Output: D:\RAW_DATA\dSPIM\E4FAD2_dw\2025_12_02_...\views\2025_12_02_..._npy2bdv_ds124_iso0p512_MD0_views.zarr

// Get the directory containing the XML
xml_dir = File.getParent(xml_path);

// Get the XML filename without extension
xml_filename = File.getName(xml_path);
xml_basename = replace(xml_filename, ".xml", "");

// Create output path: xml_dir/views/basename_views.zarr
output_dir = xml_dir + File.separator + "views";
output_zarr = output_dir + File.separator + xml_basename + "_views.zarr";

print("Output directory: " + output_dir);
print("Output zarr: " + output_zarr);

// Create output directory if it doesn't exist
if (!File.exists(output_dir)) {
    File.makeDirectory(output_dir);
    print("Created output directory: " + output_dir);
}

// ============================================
// STEP 4: Run fusion with the cropped bounding box
// ============================================
print("Step 4: Running fusion...");

run("Image Fusion", 
    "select=" + xml_path + " " +
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
    "show_advanced_block_size_options " + // the space after the parameter is fundamental
    "block_size_x=256 " +
    "block_size_y=256 " +
    "block_size_z=256 " +
    "block_size_factor_x=1 " +
    "block_size_factor_y=1 " +
    "block_size_factor_z=1");



