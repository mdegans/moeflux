//! GPU command-encoder ergonomics.
//!
//! Phase 4 of the cleanup arc. Currently houses [`pipeline_bundle!`],
//! the macro that generates the crate's pipeline-bundle structs. The
//! `ComputeEncoder` builder lands here in Phase 4a.

/// Generate a pipeline-bundle struct and its `fetch` constructor.
///
/// Every field is a `metal::ComputePipelineState` fetched by kernel
/// name; `fetch` builds the whole bundle. Replaces the hand-rolled
/// `pub struct … {} impl … { pub fn fetch … }` boilerplate that
/// recurred across the GPU kernel modules.
///
/// ```ignore
/// pipeline_bundle! {
///     /// Optional struct doc comment.
///     pub struct RmsNormBf16Pipelines {
///         sum   => "rms_norm_sum_sq",
///         /// Optional field doc comment.
///         apply => "rms_norm_apply_bf16",
///     }
/// }
/// ```
macro_rules! pipeline_bundle {
    (
        $(#[$struct_meta:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                $field:ident => $kernel:literal
            ),+ $(,)?
        }
    ) => {
        $(#[$struct_meta])*
        $vis struct $name {
            $(
                $(#[$field_meta])*
                pub $field: ::metal::ComputePipelineState,
            )+
        }

        impl $name {
            /// Fetch every kernel pipeline by name. `MetalContext`
            /// caches compiled pipelines, so repeat calls are O(1).
            pub fn fetch(
                metal: &mut $crate::riir::backend::gpu::metal::MetalContext,
            ) -> ::core::result::Result<
                Self,
                $crate::riir::backend::gpu::metal::MetalError,
            > {
                ::core::result::Result::Ok(Self {
                    $( $field: metal.pipeline($kernel)?.clone(), )+
                })
            }
        }
    };
}

pub(crate) use pipeline_bundle;
