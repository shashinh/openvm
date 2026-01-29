use proc_macro::TokenStream;

mod constraint_ext_annotation;

#[proc_macro_derive(ConstraintExtAnnotation, attributes(constraint_ext))]
pub fn constraint_ext_annotation_derive(input: TokenStream) -> TokenStream {
    constraint_ext_annotation::constraint_ext_annotation_derive(input)
}