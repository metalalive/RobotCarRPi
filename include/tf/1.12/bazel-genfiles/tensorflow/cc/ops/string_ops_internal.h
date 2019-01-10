// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_STRING_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_STRING_OPS_INTERNAL_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

/// @defgroup string_ops_internal String Ops Internal
/// @{

/// Check if the input matches the regex pattern.
///
/// The input is a string tensor of any shape. The pattern is the
/// regular expression to be matched with every element of the input tensor.
/// The boolean values (True or False) of the output tensor indicate
/// if the input matches the regex pattern provided.
///
/// The pattern follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)
///
/// Arguments:
/// * scope: A Scope object
/// * input: A string tensor of the text to be processed.
/// * pattern: The regular expression to match the input.
///
/// Returns:
/// * `Output`: A bool tensor with the same shape as `input`.
class StaticRegexFullMatch {
 public:
  StaticRegexFullMatch(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input, StringPiece pattern);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Replaces the match of pattern in input with rewrite.
///
/// It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)
///
/// Arguments:
/// * scope: A Scope object
/// * input: The text to be processed.
/// * pattern: The regular expression to match the input.
/// * rewrite: The rewrite to be applied to the matched expresion.
///
/// Optional attributes (see `Attrs`):
/// * replace_global: If True, the replacement is global, otherwise the replacement
/// is done only on the first match.
///
/// Returns:
/// * `Output`: The text after applying pattern and rewrite.
class StaticRegexReplace {
 public:
  /// Optional attribute setters for StaticRegexReplace
  struct Attrs {
    /// If True, the replacement is global, otherwise the replacement
    /// is done only on the first match.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ReplaceGlobal(bool x) {
      Attrs ret = *this;
      ret.replace_global_ = x;
      return ret;
    }

    bool replace_global_ = true;
  };
  StaticRegexReplace(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                   StringPiece pattern, StringPiece rewrite);
  StaticRegexReplace(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                   StringPiece pattern, StringPiece rewrite, const
                   StaticRegexReplace::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs ReplaceGlobal(bool x) {
    return Attrs().ReplaceGlobal(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_STRING_OPS_INTERNAL_H_
