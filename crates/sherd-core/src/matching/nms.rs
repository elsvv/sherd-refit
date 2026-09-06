//! Non-maximum suppression over poses (R §5.3): hypotheses sorted by coarse score, a later pose
//! dropped when its translation is within `nms_delta·t` of a kept one. Ties break by the lower
//! index, so the kept set does not depend on the sort's stability. Filled in in phase 1c.
