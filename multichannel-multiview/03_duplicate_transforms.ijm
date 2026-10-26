// Step 3/4: copy the registration of the reference channel to all other channels
// (matching angle/illumination/tile/timepoint).
//
// NOTE: parameter keys below are written from memory of the BigStitcher recorder
// output. Run "Duplicate Transformations" once in the GUI with the Macro Recorder
// on and paste the recorded call here if this one is rejected.
//
// Usage:
//   fiji --headless -macro 03_duplicate_transforms.ijm "<path>/dataset.xml"

// ---- settings -------------------------------------------------------------
ref_channel = 0;   // source channel (registered in step 2)
// ---------------------------------------------------------------------------

xml_path = getArgument();
xml_path = replace(xml_path, "file:", "");
print("XML: " + xml_path);

run("Duplicate Transformations",
    "apply=[One channel to other channels] " +
    "select=[" + xml_path + "] " +
    "apply_to_angle=[All angles] " +
    "apply_to_illumination=[All illuminations] " +
    "apply_to_tile=[All tiles] " +
    "apply_to_timepoint=[All Timepoints] " +
    "source=" + ref_channel + " " +
    "target=[All Channels] " +
    "duplicate_which_transformations=[Replace all transformations]");

print("Done: transformations of channel " + ref_channel + " duplicated to other channels");
