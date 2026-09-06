//! Preview images (R §11.5): a software point renderer — no GPU, no display, no font stack — as
//! a line-by-line port of the reference's, writing PNG through `image` and labelling with an
//! embedded 5×7 bitmap font. The glyphs differ from PIL's, which is accepted: previews are not
//! compared by the parity harness. Filled in in phase 1d.
