// Houdini-compatible scene unit system.
// Length, mass and time follow Houdini's default MKS setup so imported and
// authored values match Solaris / OBJ / SOP conventions.
#pragma once

namespace sol {
namespace units {

// Length: one scene unit is one metre (Houdini default).
constexpr float kMetersPerUnit = 1.0f;
constexpr const char* kLengthName = "metres";
constexpr const char* kLengthSuffix = "m";

// Mass / time (documented for parity; not all are exposed as parameters yet).
constexpr const char* kMassName = "kilograms";
constexpr const char* kTimeName = "seconds";
constexpr const char* kAngleName = "degrees";

// Camera / film conventions match Houdini (focal + aperture in millimetres,
// focus distance and transforms in scene units / metres).
constexpr const char* kFocalUnitName = "millimetres";

inline const char* lengthTooltip() {
    return "Scene units: metres (1 unit = 1 m), matching Houdini";
}

inline const char* importScaleTooltip() {
    return "Multiply imported positions into scene units (metres). "
           "Use 0.01 for centimetre assets, 0.001 for millimetre assets";
}

inline const char* focusDistanceTooltip() {
    return "Focus distance in metres (scene units). "
           "Requires F-Stop > 0 for depth of field. "
           "Use Focus Pick to set from a viewport click";
}

}  // namespace units
}  // namespace sol
