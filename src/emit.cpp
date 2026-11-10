// Copyright 2026 air2glsl contributors
// SPDX-License-Identifier: MPL-2.0
#include "emit.h"

#include <llvm/ADT/APInt.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace air2glsl {
namespace {

using namespace llvm;

/// Looks up the stage of `module` from Air's named metadata.
///
/// Air marks entry points with `!air.vertex`, `!air.fragment` or `!air.kernel`,
/// each holding `!{ptr @entry, <outputs>, <inputs>}`. Returns the stage and
/// sets `entry`.
std::string find_entry(const Module &module, Function *&entry) {
    struct Stage {
        const char *metadata;
        const char *name;
    };
    static const Stage kStages[] = {
        {"air.vertex", "vertex"},
        {"air.fragment", "fragment"},
        {"air.kernel", "kernel"},
    };
    for (const Stage &stage : kStages) {
        const NamedMDNode *named = module.getNamedMetadata(stage.metadata);
        if (named == nullptr || named->getNumOperands() == 0) continue;
        const MDNode *node = named->getOperand(0);
        if (node->getNumOperands() == 0) continue;
        if (auto *value = dyn_cast<ValueAsMetadata>(node->getOperand(0))) {
            if (auto *fn = dyn_cast<Function>(value->getValue())) {
                entry = fn;
                return stage.name;
            }
        }
    }
    return "";
}

/// std::string::starts_with is C++20; this keeps the code C++17.
inline bool starts_with(const std::string &text, const char *prefix) {
    return text.compare(0, std::strlen(prefix), prefix) == 0;
}

class Emitter {
  public:
    Emitter(const Module &module, Function *entry, std::string stage)
        : module_(module),
          entry_(entry),
          stage_(std::move(stage)),
          layout_(module.getDataLayout()) {}

    bool run(std::string &source, std::string &error);

  private:
    const Module &module_;
    Function *entry_;
    std::string stage_;
    const DataLayout &layout_;

    std::string body_;
    std::string error_;
    int tmp_ = 0;
    int phi_ = 0;

    std::map<const Value *, std::string> names_;
    std::map<const AllocaInst *, std::string> locals_;
    /// Allocas whose address is taken. They cannot be GLSL locals, because
    /// their address has to be a real value; they get a slice of a scratch
    /// buffer instead, addressed with buffer_reference.
    std::map<const AllocaInst *, uint64_t> scratch_;
    uint64_t scratch_size_ = 0;

    /// True when a value (an alloca, or a cast/GEP chain rooted at one) is used
    /// for anything other than being loaded from or stored to.
    bool address_escapes(const Value *value, std::set<const Value *> &seen) {
        if (!seen.insert(value).second) return false;
        for (const User *user : value->users()) {
            if (isa<LoadInst>(user)) continue;
            if (auto *store = dyn_cast<StoreInst>(user)) {
                if (store->getValueOperand() == value) return true;
                continue;
            }
            if (isa<GetElementPtrInst>(user)) {
                if (address_escapes(user, seen)) return true;
                continue;
            }
            if (isa<BitCastOperator>(user)) {
                // A bitcast in an access chain means the same bytes are
                // accessed as another type (packing floats into an integer
                // array, for example). With opaque pointers the cast itself
                // carries no type, so this is treated as addressing: GLSL
                // cannot pun types in a local, but a buffer_reference
                // addresses bytes, so the alloca goes to the scratch buffer.
                return true;
            }
            return true;
        }
        return false;
    }
    std::map<const BasicBlock *, int> block_ids_;
    std::vector<const BasicBlock *> blocks_;

    std::vector<Type *> struct_order_;
    std::map<Type *, std::string> struct_names_;
    /// Metal globals (`device`/`constant` variables) become base addresses the
    /// host has to supply, declared as uint64_t uniforms.
    std::vector<const GlobalVariable *> globals_;
    std::map<const GlobalVariable *, std::string> global_names_;
    std::vector<Type *> ref_order_;
    std::map<Type *, std::string> ref_names_;
    std::set<std::string> unsupported_;

    std::string fail(const std::string &message) {
        error_ = message;
        return "";
    }

    // ---- type names ------------------------------------------------------

    std::string scalar_name(Type *t) {
        if (t->isFloatTy() || t->isHalfTy()) return "float";
        if (t->isDoubleTy()) return "double";
        if (t->isIntegerTy(1)) return "bool";
        if (t->isIntegerTy(8) || t->isIntegerTy(16) || t->isIntegerTy(32)) {
            return "int";
        }
        if (t->isIntegerTy(64)) return "int64_t";
        return "";
    }

    std::string vector_name(Type *elem, unsigned count) {
        const std::string s = scalar_name(elem);
        static const char *kF[] = {"", "", "vec2", "vec3", "vec4"};
        static const char *kI[] = {"", "", "ivec2", "ivec3", "ivec4"};
        static const char *kB[] = {"", "", "bvec2", "bvec3", "bvec4"};
        static const char *kL[] = {"", "", "i64vec2", "i64vec3", "i64vec4"};
        if (count < 2 || count > 4) return "";
        if (s == "float") return kF[count];
        if (s == "int") return kI[count];
        if (s == "bool") return kB[count];
        if (s == "int64_t") return kL[count];
        return "";
    }

    /// GLSL name for a value type. Registers aggregates that need declaring.
    std::string type_name(Type *t) {
        if (t->isVoidTy()) return "void";
        // Metal passes GPU addresses around, so a pointer is a 64-bit handle;
        // GL_EXT_buffer_reference casts it to a reference when dereferenced.
        if (t->isPointerTy()) return "uint64_t";
        if (auto *vec = dyn_cast<VectorType>(t)) {
            if (vec->isScalableTy()) return fail("scalable vector");
            const unsigned count = vec->getElementCount().getFixedValue();
            if (count == 1) return scalar_name(vec->getElementType());
            const std::string name = vector_name(vec->getElementType(), count);
            if (name.empty()) return fail("unsupported vector type");
            return name;
        }
        if (auto *arr = dyn_cast<ArrayType>(t)) {
            const std::string elem = type_name(arr->getElementType());
            if (elem.empty()) return "";
            return elem + "[" + std::to_string(arr->getNumElements()) + "]";
        }
        if (auto *st = dyn_cast<StructType>(t)) {
            if (st->isOpaque()) return fail("opaque struct");
            auto it = struct_names_.find(t);
            if (it != struct_names_.end()) return it->second;
            const std::string name = "s" + std::to_string(struct_order_.size());
            struct_names_[t] = name;
            struct_order_.push_back(t);
            return name;
        }
        const std::string name = scalar_name(t);
        if (name.empty()) return fail("unsupported scalar type");
        return name;
    }

    /// True when the aggregate a texture read returns holds integer data.
    bool aggregate_is_integer(Type *type) {
        auto *st = dyn_cast<StructType>(type);
        if (st == nullptr || st->getNumElements() == 0) return false;
        Type *first = st->getElementType(0);
        Type *scalar = first->getScalarType();
        return scalar->isIntegerTy() && !scalar->isIntegerTy(1);
    }

    /// A single-member buffer_reference type for reading or writing `t` at an
    /// arbitrary GPU address.
    ///
    /// Using one such type per pointee (member always at offset 0) means the
    /// emitter never has to reproduce Apple's struct layout: the byte offsets
    /// the IR already computes stay correct, and the member is just `v0`.
    std::string ref_type(Type *t) {
        auto it = ref_names_.find(t);
        if (it != ref_names_.end()) return it->second;
        const std::string name = "Ref" + std::to_string(ref_order_.size());
        ref_names_[t] = name;
        ref_order_.push_back(t);
        return name;
    }

    // ---- values ----------------------------------------------------------

    std::string constant_expr(const Constant *c) {
        if (auto *ci = dyn_cast<ConstantInt>(c)) {
            const int64_t v = ci->getSExtValue();
            if (c->getType()->isIntegerTy(1)) return v ? "true" : "false";
            if (c->getType()->isIntegerTy(64)) return std::to_string(v) + "L";
            return std::to_string(v);
        }
        if (auto *cf = dyn_cast<ConstantFP>(c)) {
            const double value = cf->getValueAPF().convertToDouble();
            if (!std::isfinite(value)) {
                // GLSL has no NaN/Inf literals; the bit pattern is portable.
                const float f = static_cast<float>(value);
                uint32_t bits = 0;
                std::memcpy(&bits, &f, sizeof(bits));
                char text[64];
                std::snprintf(text, sizeof(text), "uintBitsToFloat(%uu)", bits);
                return std::string(text);
            }
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), "%.9g", value);
            std::string text(buffer);
            if (text.find('.') == std::string::npos &&
                text.find('e') == std::string::npos &&
                text.find("inf") == std::string::npos &&
                text.find("nan") == std::string::npos) {
                text += ".0";
            }
            if (c->getType()->isIntegerTy(64)) return text;
            return text;
        }
        if (isa<UndefValue>(c) || isa<PoisonValue>(c)) {
            // An undef aggregate still needs a correctly typed zero: it is the
            // initial value of an insertvalue/insertelement chain.
            const std::string value = default_value(c->getType());
            if (!value.empty()) return value;
            fail("undef of an unconstrustible type");
            return "";
        }
        if (isa<ConstantAggregateZero>(c)) {
            const std::string value = default_value(c->getType());
            if (!value.empty()) return value;
            fail("zero value for an unconstrustible type");
            return "";
        }
        if (auto *cd = dyn_cast<ConstantDataSequential>(c)) {
            const std::string ty = type_name(c->getType());
            if (ty.empty()) return "";
            std::string out = ty + "(";
            for (unsigned i = 0; i < cd->getNumElements(); ++i) {
                if (i) out += ", ";
                out += constant_expr(cd->getElementAsConstant(i));
            }
            return out + ")";
        }
        if (auto *cv = dyn_cast<ConstantVector>(c)) {
            const std::string ty = type_name(cv->getType());
            if (ty.empty()) return "";
            std::string out = ty + "(";
            for (unsigned i = 0; i < cv->getNumOperands(); ++i) {
                if (i) out += ", ";
                out += constant_expr(cv->getOperand(i));
            }
            return out + ")";
        }
        return fail("unsupported constant");
    }

    /// An expression usable as an array/vector index: plain integer text for
    /// constants, because GLSL does not accept an int64_t there.
    std::string index_expr(const Value *v) {
        if (isa<ConstantInt>(v)) {
            int64_t value = 0;
            if (const_index(v, value)) return std::to_string(value);
        }
        const std::string value = expr(v);
        if (value.empty()) return value;
        // GLSL wants a 32-bit int index, and Air uses 64-bit indices.
        if (v->getType()->isIntegerTy(64)) return "int(" + value + ")";
        return value;
    }

    /// GLSL expression for a non-pointer value.
    std::string expr(const Value *v) {
        if (v->getType()->isPointerTy()) {
            // Globals (including `threadgroup` globals, which Air gives a host
            // supplied address) and constant GEP expressions resolve like any
            // other pointer.
            if (isa<ConstantPointerNull>(v)) return "0L";
            if (isa<GlobalVariable>(v) || isa<GEPOperator>(v)) {
                PtrRef ref = resolve(v);
                if (!error_.empty()) return "";
                if (ref.local) {
                std::string text;
                raw_string_ostream stream(text);
                v->print(stream);
                return fail("address of local memory escapes: " +
                            stream.str().substr(0, 90));
            }
                return address_of(ref);
            }
            if (auto *constant = dyn_cast<Constant>(v)) {
                std::string text;
                raw_string_ostream stream(text);
                constant->print(stream);
                fail("unsupported constant pointer: " + stream.str().substr(0, 80));
                return "";
            }
        }
        if (auto *c = dyn_cast<Constant>(v)) {
            return constant_expr(c);
        }
        // Pointer-typed SSA values are GPU addresses; resolve them so that
        // sources (wide operand of a pointer-typed value, vector memcpy,
        // intrinsic operands) all see a uint64_t.
        if (v->getType()->isPointerTy()) {
            PtrRef ref = resolve(v);
            if (!error_.empty()) return "";
            if (ref.local) {
                std::string text;
                raw_string_ostream stream(text);
                v->print(stream);
                return fail("address of local memory escapes: " +
                            stream.str().substr(0, 90));
            }
            return address_of(ref);
        }
        const auto it = names_.find(v);
        if (it == names_.end()) {
            std::string detail = "value has no GLSL name";
            if (const auto *instruction = dyn_cast<Instruction>(v)) {
                detail += ": " + std::string(instruction->getOpcodeName());
            } else {
                detail += ": non-instruction";
            }
            return fail(detail);
        }
        return it->second;
    }

    struct PtrRef {
        bool local = false;
        std::string base;    // local variable, or uint64_t address expression
        std::string suffix;  // "[i]" / ".f" chain for locals
        std::string offset;  // byte offset for addresses, "" if zero
    };

    bool const_index(const Value *v, int64_t &out) {
        auto *ci = dyn_cast<ConstantInt>(v);
        if (ci == nullptr) return false;
        out = ci->getSExtValue();
        return true;
    }

    std::string field_name(StructType *st, unsigned index) {
        return "f" + std::to_string(index);
    }

    /// Resolves a pointer to either a named local or a uint64_t address.
    PtrRef resolve(const Value *ptr) {
        PtrRef ref;
        if (auto *alloca = dyn_cast<AllocaInst>(ptr)) {
            const auto scratch = scratch_.find(alloca);
            if (scratch != scratch_.end()) {
                ref.local = false;
                ref.base = "g_globals.scratch";
                if (scratch->second != 0) {
                    ref.offset = std::to_string(scratch->second) + "L";
                }
                return ref;
            }
            ref.local = true;
            ref.base = locals_[alloca];
            return ref;
        }
        if (auto *gep = dyn_cast<GEPOperator>(ptr)) {
            PtrRef base = resolve(gep->getPointerOperand());
            if (base.local) {
                ref = base;
                Type *cur = gep->getSourceElementType();
                bool first = true;
                for (const Use &index : gep->indices()) {
                    int64_t value = 0;
                    const bool is_constant = const_index(index.get(), value);
                    if (first) {
                        if (!is_constant) {
                            fail("dynamic first index on a local");
                            return ref;
                        }
                        // The first index steps over whole objects of the
                        // source type, so a non-zero value means this pointer
                        // is not simply addressing a local.
                        first = false;
                        if (value != 0) {
                            fail("non-zero first GEP index on a local");
                            return ref;
                        }
                        continue;
                    }
                    if (auto *st = dyn_cast<StructType>(cur)) {
                        if (!is_constant || value < 0 ||
                            static_cast<unsigned>(value) >= st->getNumElements()) {
                            fail("dynamic struct index on a local");
                            return ref;
                        }
                        ref.suffix += "." + field_name(st, (unsigned)value);
                        cur = st->getElementType((unsigned)value);
                        continue;
                    }
                    Type *element = nullptr;
                    if (auto *at = dyn_cast<ArrayType>(cur)) {
                        element = at->getElementType();
                    } else if (auto *vt = dyn_cast<VectorType>(cur)) {
                        element = vt->getElementType();
                    } else {
                        fail("index into a non-aggregate");
                        return ref;
                    }
                    // GLSL happily indexes local arrays and vectors with a
                    // runtime value, so dynamic indices are fine here.
                    if (is_constant) {
                        ref.suffix += "[" + std::to_string(value) + "]";
                    } else {
                        const std::string operand = index_expr(index.get());
                        if (operand.empty()) return ref;
                        ref.suffix += "[" + operand + "]";
                    }
                    cur = element;
                }
                return ref;
            }
            APInt bytes(layout_.getIndexSizeInBits(gep->getPointerAddressSpace()), 0);
            if (gep->accumulateConstantOffset(layout_, bytes)) {
                ref = base;
                const int64_t offset = bytes.getSExtValue();
                if (offset != 0) ref.offset = std::to_string(offset) + "L";
                return ref;
            }
            // A dynamic index (an array of resource descriptors, for example)
            // turns into a runtime byte offset: this is exactly why the
            // translation target is Vulkan GLSL with buffer_reference, where
            // address arithmetic is expressible.
            ref = base;
            {
                std::string terms;
                uint64_t constant = 0;
                Type *cur = gep->getSourceElementType();
                bool first_index = true;
                bool failed = false;
                for (const Use &index : gep->indices()) {
                    int64_t value = 0;
                    const bool is_constant = const_index(index.get(), value);
                    if (first_index) {
                        first_index = false;
                        const uint64_t size = layout_.getTypeAllocSize(cur);
                        if (is_constant) {
                            constant += static_cast<uint64_t>(value) * size;
                        } else {
                            const std::string operand = expr(index.get());
                            if (operand.empty()) return ref;
                            terms += (terms.empty() ? "" : " + ") + operand +
                                     " * " + std::to_string(size) + "L";
                        }
                        continue;
                    }
                    if (auto *st = dyn_cast<StructType>(cur)) {
                        if (!is_constant || value < 0 ||
                            static_cast<unsigned>(value) >= st->getNumElements()) {
                            fail("dynamic struct index in an address");
                            failed = true;
                            break;
                        }
                        constant += layout_.getStructLayout(st)->getElementOffset(
                            static_cast<unsigned>(value));
                        cur = st->getElementType(static_cast<unsigned>(value));
                        continue;
                    }
                    Type *element = nullptr;
                    if (auto *at = dyn_cast<ArrayType>(cur)) {
                        element = at->getElementType();
                    } else if (auto *vt = dyn_cast<VectorType>(cur)) {
                        element = vt->getElementType();
                    } else {
                        fail("index into a non-aggregate address");
                        failed = true;
                        break;
                    }
                    const uint64_t size = layout_.getTypeAllocSize(element);
                    if (is_constant) {
                        constant += static_cast<uint64_t>(value) * size;
                    } else {
                        const std::string operand = expr(index.get());
                        if (operand.empty()) return ref;
                        terms += (terms.empty() ? "" : " + ") + operand + " * " +
                                 std::to_string(size) + "L";
                    }
                    cur = element;
                }
                if (failed) return ref;
                if (terms.empty()) {
                    if (constant != 0) {
                        ref.offset = std::to_string(constant) + "L";
                    }
                } else {
                    ref.offset = "(" + terms + " + " +
                                 std::to_string(constant) + "L)";
                }
            }
            return ref;
        }
        if (auto *global = dyn_cast<GlobalVariable>(ptr)) {
            ref.local = false;
            ref.base = global_address(global);
            return ref;
        }
        if (auto *cast = dyn_cast<BitCastOperator>(ptr)) {
            return resolve(cast->getOperand(0));
        }
        if (auto *cast = dyn_cast<AddrSpaceCastOperator>(ptr)) {
            return resolve(cast->getOperand(0));
        }
        if (auto *ci = dyn_cast<CastInst>(ptr)) {
            if (isa<IntToPtrInst>(ci)) return resolve(ci->getOperand(0));
        }
        ref.local = false;
        // A pointer that is a plain value (a load from an argument buffer, a
        // function argument, ...) is already a uint64_t GPU address.
        const auto it = names_.find(ptr);
        if (it == names_.end()) {
            std::string detail = "pointer value has no GLSL name: ";
            std::string text;
            raw_string_ostream stream(text);
            ptr->printAsOperand(stream, false);
            detail += stream.str();
            fail(detail);
            return ref;
        }
        ref.base = it->second;
        return ref;
    }

    /// Name of the uniform that carries the address of a Metal global.
    std::string global_address(const GlobalVariable *global) {
        auto it = global_names_.find(global);
        if (it != global_names_.end()) return it->second;
        const std::string name =
            "g_globals.ga[" + std::to_string(globals_.size()) + "]";
        global_names_[global] = name;
        globals_.push_back(global);
        return name;
    }

    std::string address_of(const PtrRef &ref) {
        if (ref.offset.empty()) return ref.base;
        return "(" + ref.base + " + " + ref.offset + ")";
    }

    /// Reads a value of type `type` through `ref`.
    std::string load_through(const PtrRef &ref, Type *type) {
        if (ref.local) return "(" + ref.base + ref.suffix + ")";
        const std::string ty = type_name(type);
        if (ty.empty()) return "";
        return ref_type(type) + "(" + address_of(ref) + ").v0";
    }

    // ---- instruction emission -------------------------------------------

    void line(const std::string &text) { body_ += "    " + text + "\n"; }

    std::string cast_to(Type *from, Type *to, const std::string &value) {
        const std::string target = type_name(to);
        if (target.empty()) return "";
        if (from == to) return value;
        if (from->isPointerTy() || to->isPointerTy()) {
            // Pointers are all uint64_t handles here.
            return value;
        }
        if (from->isFloatingPointTy() && to->isIntegerTy()) {
            return "int(" + value + ")";
        }
        if (from->isIntegerTy() && to->isFloatingPointTy()) {
            return "float(" + value + ")";
        }
        if (from->isFloatingPointTy() && to->isFloatingPointTy()) {
            // Half and double both become float in the emitted GLSL, so this
            // is only ever a widening/narrowing conversion to the target type.
            return target + "(" + value + ")";
        }
        return target + "(" + value + ")";
    }

    std::string bitcast(Type *from, Type *to, const std::string &value) {
        if (from == to) return value;
        if (from->isPointerTy() || to->isPointerTy()) return value;
        // Half to float and friends are plain conversions in GLSL.
        if (from->getScalarType()->isFloatingPointTy() &&
            to->getScalarType()->isFloatingPointTy()) {
            return value;
        }
        const unsigned from_bits = layout_.getTypeSizeInBits(from);
        const unsigned to_bits = layout_.getTypeSizeInBits(to);
        if (from_bits != to_bits) return fail("bitcast of mismatched sizes");
        const bool from_fp = from->getScalarType()->isFloatingPointTy();
        const bool to_fp = to->getScalarType()->isFloatingPointTy();
        const bool to_signed = to->getScalarType()->isIntegerTy() &&
                               !to->getScalarType()->isIntegerTy(1);
        if (from_fp && to_fp) return value;
        if (from_fp && !to_fp) {
            return (to_signed ? "floatBitsToInt(" : "floatBitsToUint(") + value +
                   ")";
        }
        if (!from_fp && to_fp) {
            return (from->getScalarType()->isIntegerTy(1)
                        ? std::string("float(") + value + ")"
                        : std::string("intBitsToFloat(") + value + ")");
        }
        return value;
    }

    std::string binary_op(unsigned opcode, Type *type, const std::string &a,
                          const std::string &b) {
        const char *op = nullptr;
        switch (opcode) {
            case Instruction::Add:
            case Instruction::FAdd: op = "+"; break;
            case Instruction::Sub:
            case Instruction::FSub: op = "-"; break;
            case Instruction::Mul:
            case Instruction::FMul: op = "*"; break;
            case Instruction::UDiv:
            case Instruction::SDiv:
            case Instruction::FDiv: op = "/"; break;
            case Instruction::URem:
            case Instruction::SRem:
            case Instruction::FRem: op = "%"; break;
            case Instruction::And: op = "&"; break;
            case Instruction::Or: op = "|"; break;
            case Instruction::Xor: op = "^"; break;
            case Instruction::Shl: op = "<<"; break;
            case Instruction::LShr:
            case Instruction::AShr: op = ">>"; break;
            default: return fail("unsupported binary operator");
        }
        if (type->isIntegerTy(1)) {
            // i1 arithmetic in the IR is logical in GLSL.
            std::string logical = op;
            if (std::strcmp(op, "&") == 0) logical = "&&";
            if (std::strcmp(op, "|") == 0) logical = "||";
            if (std::strcmp(op, "^") == 0) logical = "!=";
            return "(" + a + " " + logical + " " + b + ")";
        }
        return "(" + a + " " + op + " " + b + ")";
    }

    std::string compare_op(CmpInst::Predicate predicate,
                           const std::string &a, const std::string &b) {
        const char *op = nullptr;
        switch (predicate) {
            case CmpInst::ICMP_EQ:
            case CmpInst::FCMP_OEQ:
            case CmpInst::FCMP_UEQ: op = "=="; break;
            case CmpInst::ICMP_NE:
            case CmpInst::FCMP_UNE:
            case CmpInst::FCMP_ONE: op = "!="; break;
            case CmpInst::ICMP_ULT:
            case CmpInst::ICMP_SLT:
            case CmpInst::FCMP_OLT:
            case CmpInst::FCMP_ULT: op = "<"; break;
            case CmpInst::ICMP_ULE:
            case CmpInst::ICMP_SLE:
            case CmpInst::FCMP_OLE:
            case CmpInst::FCMP_ULE: op = "<="; break;
            case CmpInst::ICMP_UGT:
            case CmpInst::ICMP_SGT:
            case CmpInst::FCMP_OGT:
            case CmpInst::FCMP_UGT: op = ">"; break;
            case CmpInst::ICMP_UGE:
            case CmpInst::ICMP_SGE:
            case CmpInst::FCMP_OGE:
            case CmpInst::FCMP_UGE: op = ">="; break;
            default: return fail("unsupported comparison predicate");
        }
        return "(" + a + " " + op + " " + b + ")";
    }

    /// Maps an Air global atomic onto a GLSL atomic on a buffer variable.
    std::string atomic_call(const CallInst &call, const std::string &name) {
        Type *element = call.getType();
        if (!element->isIntegerTy()) {
            return fail("atomic on a non-integer type: " + name);
        }
        std::string operation;
        bool negate = false;
        if (name.find(".add.") != std::string::npos) {
            operation = "atomicAdd";
        } else if (name.find(".sub.") != std::string::npos) {
            operation = "atomicAdd";
            negate = true;
        } else if (name.find(".and.") != std::string::npos) {
            operation = "atomicAnd";
        } else if (name.find(".or.") != std::string::npos) {
            operation = "atomicOr";
        } else if (name.find(".xor.") != std::string::npos) {
            operation = "atomicXor";
        } else if (name.find(".min.") != std::string::npos) {
            operation = "atomicMin";
        } else if (name.find(".max.") != std::string::npos) {
            operation = "atomicMax";
        } else if (name.find(".xchg.") != std::string::npos) {
            operation = "atomicExchange";
        } else if (name.find(".cmpxchg.") != std::string::npos) {
            // (ptr, ptr comparand, new value, ...): the comparand arrives as a
            // pointer (often to a local), so it is loaded rather than
            // addressed, which also keeps the local from escaping.
            const std::string comparand = load_through(
                resolve(call.getArgOperand(1)), element);
            if (comparand.empty()) return "";
            return "atomicCompSwap(" + ref_type(element) + "(" +
                   expr(call.getArgOperand(0)) + ").v0, " + comparand + ", " +
                   expr(call.getArgOperand(2)) + ")";
        } else {
            return fail("unsupported atomic: " + name);
        }
        const std::string value = expr(call.getArgOperand(1));
        const std::string operand = negate ? "(-" + value + ")" : value;
        return operation + "(" + ref_type(element) + "(" +
               expr(call.getArgOperand(0)) + ").v0, " + operand + ")";
    }

    std::string emit_intrinsic(const CallInst &call) {
        Function *callee = call.getCalledFunction();
        if (callee == nullptr) return fail("indirect call");
        const std::string name = callee->getName().str();
        auto arg = [&](unsigned i) -> std::string {
            return expr(call.getArgOperand(i));
        };
        const unsigned n = call.arg_size();

        // Unary and binary Air intrinsics that map onto GLSL built-ins.
        struct Mapping {
            const char *air;
            const char *glsl;
            unsigned operands;
        };
        static const Mapping kMap[] = {
            {"air.fast_fmax", "max", 2},      {"air.fast_fmin", "min", 2},
            {"air.fast_fabs", "abs", 1},      {"air.fast_sqrt", "sqrt", 1},
            {"air.fast_rsqrt", "inversesqrt", 1},
            {"air.fast_exp2", "exp2", 1},     {"air.fast_log2", "log2", 1},
            {"air.fast_rint", "roundEven", 1},
            {"air.fast_floor", "floor", 1},   {"air.fast_ceil", "ceil", 1},
            {"air.fast_trunc", "trunc", 1},   {"air.fast_sin", "sin", 1},
            {"air.fast_cos", "cos", 1},       {"air.fast_tan", "tan", 1},
            {"air.fast_asin", "asin", 1},     {"air.fast_acos", "acos", 1},
            {"air.fast_atan", "atan", 1},     {"air.fast_pow", "pow", 2},
            {"air.fma", "fma", 3},            {"air.dot", "dot", 2},
            {"air.min", "min", 2},            {"air.max", "max", 2},
            {"air.fast_fract", "fract", 1},   {"air.fract", "fract", 1},
            {"air.dfdx", "dFdx", 1},          {"air.dfdy", "dFdy", 1},
            {"air.dfdx_fine", "dFdxFine", 1}, {"air.dfdy_fine", "dFdyFine", 1},
            {"air.fwidth", "fwidth", 1},
        };
        for (const Mapping &mapping : kMap) {
            const size_t length = std::strlen(mapping.air);
            if (name.compare(0, length, mapping.air) != 0) continue;
            // The IR names carry a type suffix (".f32", ".v3f32"); require one
            // so that "air.fast_fmax" cannot match "air.fast_fmax_other".
            if (name.size() > length && name[length] != '.') continue;
            if (n < mapping.operands) return fail("intrinsic arity");
            std::string out = std::string(mapping.glsl) + "(";
            for (unsigned i = 0; i < mapping.operands; ++i) {
                if (i) out += ", ";
                out += arg(i);
            }
            return out + ")";
        }
        if (starts_with(name, "air.fast_saturate") ||
            starts_with(name, "air.saturate")) {
            std::string zero = "0.0";
            std::string one = "1.0";
            if (call.getArgOperand(0)->getType()->isVectorTy()) {
                zero = "0.0";
                one = "1.0";
            }
            return "clamp(" + arg(0) + ", " + zero + ", " + one + ")";
        }
        if (starts_with(name, "llvm.")) {
            // A handful of LLVM maths intrinsics show up because Metal's
            // compiler lowers some Air operations onto them.
            struct LlvmMapping {
                const char *name;
                const char *glsl;
            };
            static const LlvmMapping kLlvm[] = {
                {"llvm.fmuladd", "fma"},  {"llvm.fabs", "abs"},
                {"llvm.sqrt", "sqrt"},    {"llvm.minnum", "min"},
                {"llvm.maxnum", "max"},   {"llvm.floor", "floor"},
                {"llvm.ceil", "ceil"},    {"llvm.trunc", "trunc"},
                {"llvm.rint", "roundEven"}, {"llvm.round", "round"},
                {"llvm.smax", "max"},     {"llvm.smin", "min"},
                {"llvm.umax", "max"},     {"llvm.umin", "min"},
                {"llvm.exp2", "exp2"},    {"llvm.log2", "log2"},
                {"llvm.fma", "fma"},      {"llvm.ctpop", "bitCount"},
                {"llvm.ctlz", "findMSB"},
            };
            for (const LlvmMapping &mapping : kLlvm) {
                if (!starts_with(name, mapping.name)) continue;
                std::string out = std::string(mapping.glsl) + "(";
                for (unsigned i = 0; i < n; ++i) {
                    if (i) out += ", ";
                    out += arg(i);
                }
                return out + ")";
            }
            unsupported_.insert(name);
            return fail("unsupported llvm intrinsic: " + name);
        }
        if (starts_with(name, "air.convert.")) {
            return "float(" + arg(0) + ")";
        }
        if (name == "air.simdgroup.barrier" || name == "air.barrier" ||
            name == "air.simdgroup_barrier" || name == "air.memory_barrier") {
            return "barrier()";
        }
        if (name == "air.discard_fragment" || name == "air.discard") {
            return "discard";
        }
        if (name == "irconverter_discard") {
            if (auto *ci = dyn_cast<ConstantInt>(call.getArgOperand(0))) {
                if (ci->isOne()) return "discard";
            }
            return "0";
        }
        // Texture reads return an aggregate {<4 x float>, i8}; the i8 is a
        // status code that Metal exposes but shaders rarely look at, so it is
        // constructed as 0.
        if (starts_with(name, "air.read_texture_2d_array.") && n >= 3) {
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            // The integer variants read from the integer image array; a
            // float image would not match the aggregate's ivec4 field.
            const char *array =
                aggregate_is_integer(call.getType()) ? "g_images_int"
                                                     : "g_images";
            return aggregate + "(imageLoad(" + array +
                   "[nonuniformEXT(int(" + arg(0) + " & 0xffffu))], ivec3(" +
                   arg(1) + ", " + arg(2) + ")), 0)";
        }
        if (starts_with(name, "air.read_texture_3d.") && n >= 2) {
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            const char *array = aggregate_is_integer(call.getType())
                                    ? "g_images3d_int"
                                    : "g_images3d";
            return aggregate + "(imageLoad(" + array +
                   "[nonuniformEXT(int(" + arg(0) + " & 0xffffu))], ivec3(" +
                   arg(1) + ")), 0)";
        }
        if (name == "air.write_texture_2d_array.v4f32" && n >= 4) {
            helpers_.insert("write2darray");
            return "air_write_2d_array(" + arg(0) + ", ivec3(" + arg(1) + ", " +
                   arg(2) + "), " + arg(3) + ")";
        }
        if (name == "air.write_texture_3d.v4f32" && n >= 3) {
            helpers_.insert("write3d");
            return "air_write_3d(" + arg(0) + ", ivec3(" + arg(1) + "), " +
                   arg(2) + ")";
        }
        if (starts_with(name, "air.get_width_texture_2d_array")) {
            return "textureSize(g_tex_2darray[nonuniformEXT(int(" + arg(0) +
                   " & 0xffffu))], 0).x";
        }
        if (starts_with(name, "air.get_height_texture_3d")) {
            return "textureSize(g_images3d[nonuniformEXT(int(" + arg(0) +
                   " & 0xffffu))], 0).y";
        }
        if (starts_with(name, "air.fence_texture_")) {
            // A resource fence in Metal orders writes to a texture before the
            // next draw. memoryBarrier is the conservative GLSL equivalent.
            return "memoryBarrier()";
        }
        if (name == "air.gather_texture_2d_array.u.v4i32" && n >= 4) {
            helpers_.insert("gather2darray_u");
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            return aggregate + "(air_gather_2d_array_u(" + arg(0) + ", " +
                   arg(1) + ", " + arg(2) + ", " + arg(3) + "), 0)";
        }
        if (name == "air.sample_texture_cube_array.v4f32" && n >= 7) {
            helpers_.insert("samplecube");
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            return aggregate + "(air_sample_cube_array(" + arg(0) + ", " +
                   arg(1) + ", " + arg(2) + ", " + arg(3) + "), 0)";
        }
        if (name == "air.write_texture_2d_array.u.v4i32" && n >= 4) {
            helpers_.insert("write2darray_int");
            return "air_write_2d_array_int(" + arg(0) + ", ivec3(" + arg(1) +
                   ", " + arg(2) + "), " + arg(3) + ")";
        }
        if (starts_with(name, "air.read_texture_buffer_1d.") && n >= 2) {
            helpers_.insert("readbuffer");
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            if (aggregate_is_integer(call.getType())) {
                return aggregate + "(ivec4(texelFetch(g_ubuffer[nonuniformEXT(" +
                       "int(" + arg(0) + " & 0xffffu))], int(" + arg(1) +
                       "))), 0)";
            }
            return aggregate + "(texelFetch(g_tex_buffer[nonuniformEXT(int(" +
                   arg(0) + " & 0xffffu))], int(" + arg(1) + ")), 0)";
        }
        if (name == "air.read_texture_2d_ms_array.v4f32" && n >= 4) {
            helpers_.insert("readms");
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            return aggregate + "(air_read_2d_ms_array(" + arg(0) + ", ivec3(" +
                   arg(1) + ", " + arg(2) + "), " + arg(3) + "), 0)";
        }
        if (starts_with(name, "air.get_height_texture")) {
            return "textureSize(g_tex_2darray[nonuniformEXT(int(" + arg(0) +
                   " & 0xffffu))], 0).y";
        }
        if (starts_with(name, "air.wg.barrier") ||
            starts_with(name, "air.wg_barrier")) {
            return "barrier()";
        }
        if (starts_with(name, "air.ctz")) {
            return "findLSB(" + arg(0) + ")";
        }
        if (starts_with(name, "air.clz")) {
            return "findMSB(" + arg(0) + ")";
        }
        if (starts_with(name, "air.popcount")) {
            return "bitCount(" + arg(0) + ")";
        }
        if (starts_with(name, "air.get_num_mip_levels_texture")) {
            return "textureQueryLevels(g_tex_2darray[nonuniformEXT(int(" +
                   arg(0) + " & 0xffffu))])";
        }
        if (name == "air.write_texture_buffer_1d.u.v4i32" && n >= 3) {
            helpers_.insert("writebuffer");
            return "air_write_buffer(" + arg(0) + ", " + arg(1) + ", " +
                   arg(2) + ")";
        }
        // ---- depth-compare sampling ------------------------------------
        // Shadow sampling needs a combined shadow sampler: GLSL cannot build
        // one from a separate texture and sampler, so the app's sampler state
        // is approximated by the bound shadow sampler.
        if (name == "air.sample_compare_depth_2d_array.f32" && n >= 6) {
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            return aggregate + "(texture(g_shadow_2darray[nonuniformEXT(int(" +
                   arg(0) + " & 0xffffu))], vec4(" + arg(3) + ", " + arg(4) +
                   ", " + arg(5) + ")), 0)";
        }
        if (name == "air.gather_compare_depth_2d_array.f32" && n >= 6) {
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            // The shadow form of textureGather takes the coordinate (uv plus
            // layer, as a vec3) and the reference value separately.
            return aggregate + "(textureGather(g_shadow_2darray[nonuniformEXT(" +
                   "int(" + arg(0) + " & 0xffffu))], vec3(" + arg(3) + ", " +
                   "float(" + arg(4) + ")), " + arg(5) + "), 0)";
        }

        // ---- explicit gradients ----------------------------------------
        if (name == "air.sample_texture_2d_array_grad.v4f32" && n >= 6) {
            helpers_.insert("sample2darraygrad");
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            return aggregate + "(air_sample_2d_array_grad(" + arg(0) + ", " +
                   arg(1) + ", " + arg(2) + ", " + arg(3) + ", " + arg(4) +
                   ", " + arg(5) + "), 0)";
        }
        if (name == "air.calculate_clamped_lod_texture_2d_array" && n >= 4) {
            // textureQueryLod's .y is the level of detail clamped to the
            // texture's own limits, which is what this Air intrinsic returns.
            // For an array sampler, textureQueryLod takes only the coordinate
            // (vec2); the layer is not part of its argument.
            return std::string("textureQueryLod("
                               "g_tex_2darray_combined[nonuniformEXT(int(") +
                   arg(0) + " & 0xffffu))], vec2(" + arg(2) + ")).y";
        }

        // ---- remaining texture shapes ----------------------------------
        if (name == "air.write_texture_buffer_1d.v4f32" && n >= 3) {
            helpers_.insert("writebufferf");
            return "air_write_buffer_f(" + arg(0) + ", int(" + arg(1) + "), " +
                   arg(2) + ")";
        }
        if (starts_with(name, "air.read_texture_2d_ms_array.") && n >= 4) {
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            const char *array = aggregate_is_integer(call.getType())
                                    ? "g_images_ms_int"
                                    : "g_images_ms";
            return aggregate + "(imageLoad(" + array + "[nonuniformEXT(int(" +
                   arg(0) + " & 0xffffu))], ivec3(" + arg(1) + ", " + arg(2) +
                   "), " + arg(3) + "), 0)";
        }
        if (starts_with(name, "air.get_width_texture_3d")) {
            return "textureSize(g_tex_3d[nonuniformEXT(int(" + arg(0) +
                   " & 0xffffu))], 0).x";
        }
        if (starts_with(name, "air.get_height_texture_3d")) {
            return "textureSize(g_tex_3d[nonuniformEXT(int(" + arg(0) +
                   " & 0xffffu))], 0).y";
        }
        if (starts_with(name, "air.get_array_size_texture_2d_array")) {
            return "textureSize(g_tex_2darray[nonuniformEXT(int(" + arg(0) +
                   " & 0xffffu))], 0).z";
        }

        // ---- atomics on global memory ----------------------------------
        // Air models a global atomic as (address, ...args); GLSL atomics work
        // on a buffer variable, which a buffer_reference member is.
        if (starts_with(name, "air.atomic.")) {
            return atomic_call(call, name);
        }
        if (starts_with(name, "air.fast_clamp")) {
            return "clamp(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")";
        }
        if (starts_with(name, "air.reverse_bits")) {
            return "bitfieldReverse(" + arg(0) + ")";
        }
        if (name == "air.sample_texture_2d_array.v4f32" && n >= 9) {
            helpers_.insert("sample2darray");
            // texture, sampler, coord, array index, ..., offset, lod, bias
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            return aggregate + "(air_sample_2d_array(" + arg(0) + ", " +
                   arg(1) + ", " + arg(2) + ", " + arg(3) + ", " + arg(7) +
                   ", " + arg(8) + "), 0)";
        }
        if (name == "air.sample_texture_3d.v4f32" && n >= 9) {
            helpers_.insert("sample3d");
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            return aggregate + "(air_sample_3d(" + arg(0) + ", " + arg(1) +
                   ", " + arg(2) + ", " + arg(6) + ", " + arg(7) + "), 0)";
        }
        if (name == "air.gather_texture_2d_array.v4f32" && n >= 4) {
            helpers_.insert("gather2darray");
            const std::string aggregate = type_name(call.getType());
            if (aggregate.empty()) return "";
            return aggregate + "(air_gather_2d_array(" + arg(0) + ", " +
                   arg(1) + ", " + arg(2) + ", " + arg(3) + "), 0)";
        }

        if (name == "air.write_texture_2d_array.v4f32" && n >= 4) {
            helpers_.insert("write2darray");
            return "air_write_2d_array(" + arg(0) + ", " + arg(1) + ", " +
                   arg(2) + ", " + arg(3) + ")";
        }
        unsupported_.insert(name);
        return fail("unsupported intrinsic: " + name);
    }

    std::set<std::string> helpers_;

    void emit_predecessor_edge(const BasicBlock *from, const BasicBlock *to) {
        for (const PHINode &phi : to->phis()) {
            const Value *incoming = phi.getIncomingValueForBlock(from);
            const std::string value = expr(incoming);
            line(names_[&phi] + " = " + value + ";");
        }
    }

    /// The statements a jump from `from` to `to` needs: the phi assignments
    /// for that edge, then the dispatcher update.
    std::string edge_body(const BasicBlock *from, const BasicBlock *to) {
        const std::string saved = body_;
        body_.clear();
        emit_predecessor_edge(from, to);
        line("_pc = " + std::to_string(block_ids_.at(to)) + ";");
        const std::string result = body_;
        body_ = saved;
        return result;
    }

    bool emit_block(const BasicBlock &block, int id) {
        body_ += "        case " + std::to_string(id) + ": {\n";
        for (const Instruction &instruction : block) {
            if (!emit_instruction(instruction, block)) return false;
            if (!error_.empty()) return false;
        }
        body_ += "            break;\n        }\n";
        return true;
    }

    bool emit_instruction(const Instruction &instruction,
                          const BasicBlock &block) {
        const std::string tmp = names_.count(&instruction) ? names_[&instruction] : std::string();
        if (isa<PHINode>(instruction)) {
            // Phi values are assigned by the predecessor block; nothing to do.
            return true;
        }
        if (isa<AllocaInst>(instruction) || isa<GetElementPtrInst>(instruction) ||
            isa<DbgInfoIntrinsic>(instruction)) {
            return true;
        }
        if (auto *intrinsic = dyn_cast<IntrinsicInst>(&instruction)) {
            const StringRef name = intrinsic->getCalledFunction()->getName();
            static const char *kIgnored[] = {
                "llvm.lifetime", "llvm.dbg", "llvm.assume",
                "llvm.sideeffect", "llvm.donothing", "llvm.prefetch",
                "llvm.memcpy", "llvm.memset", "llvm.experimental.noalias",
                "llvm.invariant", "llvm.launder", "llvm.strip",
            };
            for (const char *prefix : kIgnored) {
                if (name.starts_with(prefix)) return true;
            }
        }
        if (auto *load = dyn_cast<LoadInst>(&instruction)) {
            const std::string value =
                load_through(resolve(load->getPointerOperand()), load->getType());
            if (value.empty()) return false;
            line(tmp + " = " + value +
                 ";");
            return true;
        }
        if (auto *store = dyn_cast<StoreInst>(&instruction)) {
            const Value *value = store->getValueOperand();
            PtrRef ref = resolve(store->getPointerOperand());
            if (!error_.empty()) return false;
            if (ref.local) {
                line("    " + std::string(ref.base) + ref.suffix + " = " +
                     expr(value) + ";");
            } else {
                line("    " + ref_type(value->getType()) + "(" +
                     address_of(ref) + ").v0 = " + expr(value) + ";");
            }
            return true;
        }
        if (auto *binary = dyn_cast<BinaryOperator>(&instruction)) {
            const std::string value =
                binary_op(binary->getOpcode(), binary->getType(),
                          expr(binary->getOperand(0)),
                          expr(binary->getOperand(1)));
            if (value.empty()) return false;
            line(tmp + " = " + value +
                 ";");
            return true;
        }
        if (auto *compare = dyn_cast<CmpInst>(&instruction)) {
            const std::string value =
                compare_op(compare->getPredicate(), expr(compare->getOperand(0)),
                        expr(compare->getOperand(1)));
            if (value.empty()) return false;
            line(tmp + " = " +
                 value + ";");
            return true;
        }
        if (auto *select = dyn_cast<SelectInst>(&instruction)) {
            const bool vector_condition =
                select->getCondition()->getType()->isVectorTy();
            // GLSL only allows a scalar `?:` condition; a vector condition
            // needs mix(), which selects componentwise.
            const std::string value =
                vector_condition
                    ? "mix(" + expr(select->getTrueValue()) + ", " +
                          expr(select->getFalseValue()) + ", " +
                          expr(select->getCondition()) + ")"
                    : "(" + expr(select->getCondition()) + " ? " +
                          expr(select->getTrueValue()) + " : " +
                          expr(select->getFalseValue()) + ")";
            line(tmp + " = " +
                 value + ";");
            return true;
        }
        if (auto *cast = dyn_cast<CastInst>(&instruction)) {
            std::string value;
            if (isa<BitCastInst>(cast)) {
                value = bitcast(cast->getSrcTy(), cast->getDestTy(),
                                expr(cast->getOperand(0)));
            } else if (isa<PtrToIntInst>(cast) || isa<IntToPtrInst>(cast) ||
                       isa<AddrSpaceCastInst>(cast)) {
                value = expr(cast->getOperand(0));
            } else {
                value = cast_to(cast->getSrcTy(), cast->getDestTy(),
                                expr(cast->getOperand(0)));
            }
            if (value.empty()) return false;
            line(tmp + " = " + value +
                 ";");
            return true;
        }
        if (auto *extract = dyn_cast<ExtractElementInst>(&instruction)) {
            line(tmp + " = " +
                 expr(extract->getVectorOperand()) + "[" +
                 index_expr(extract->getIndexOperand()) + "];");
            return true;
        }
        if (auto *insert = dyn_cast<InsertElementInst>(&instruction)) {
            line(tmp + " = " +
                 expr(insert->getOperand(0)) + ";");
            line(tmp + "[" + index_expr(insert->getOperand(2)) + "] = " +
                 expr(insert->getOperand(1)) + ";");
            return true;
        }
        if (auto *shuffle = dyn_cast<ShuffleVectorInst>(&instruction)) {
            const std::string ty = type_name(shuffle->getType());
            if (ty.empty()) return false;
            line(tmp + " = " + ty + "(0);");
            for (int i = 0; i < shuffle->getType()->getPrimitiveSizeInBits() /
                                    shuffle->getType()->getScalarSizeInBits();
                 ++i) {
                const int source = shuffle->getMaskValue(i);
                if (source < 0) continue;
                const Value *vector =
                    source < static_cast<int>(
                                 shuffle->getOperand(0)
                                     ->getType()
                                     ->getPrimitiveSizeInBits() /
                                 shuffle->getOperand(0)
                                     ->getType()
                                     ->getScalarSizeInBits())
                        ? shuffle->getOperand(0)
                        : shuffle->getOperand(1);
                const int index =
                    source < static_cast<int>(
                                 shuffle->getOperand(0)
                                     ->getType()
                                     ->getPrimitiveSizeInBits() /
                                 shuffle->getOperand(0)
                                     ->getType()
                                     ->getScalarSizeInBits())
                        ? source
                        : source - static_cast<int>(
                                       shuffle->getOperand(0)
                                           ->getType()
                                           ->getPrimitiveSizeInBits() /
                                       shuffle->getOperand(0)
                                           ->getType()
                                           ->getScalarSizeInBits());
                line("    " + tmp + "[" + std::to_string(i) + "] = " +
                     expr(vector) + "[" + std::to_string(index) + "];");
            }
            return true;
        }
        if (auto *call = dyn_cast<CallInst>(&instruction)) {
            const std::string value = emit_intrinsic(*call);
            if (value.empty()) return false;
            if (!call->getType()->isVoidTy()) {
                line(tmp + " = " +
                     value + ";");
            } else if (value != "0") {
                line("    " + value + ";");
            }
            return true;
        }
        if (auto *branch = dyn_cast<BranchInst>(&instruction)) {
            if (branch->isUnconditional()) {
                body_ += edge_body(&block, branch->getSuccessor(0));
            } else {
                const std::string condition = expr(branch->getCondition());
                const std::string then_body =
                    edge_body(&block, branch->getSuccessor(0));
                const std::string else_body =
                    edge_body(&block, branch->getSuccessor(1));
                body_ += "        if (" + condition + ") {\n" + then_body +
                         "        } else {\n" + else_body + "        }\n";
            }
            return true;
        }
        if (auto *ret = dyn_cast<ReturnInst>(&instruction)) {
            if (Value *value = ret->getReturnValue()) {
                names_[ret] = "_result";
                line("    _result = " + expr(value) + ";");
            }
            line("    _pc = -1;");
            return true;
        }
        if (auto *rmw = dyn_cast<AtomicRMWInst>(&instruction)) {
            // The result of an atomicrmw is the value before the operation.
            const char *operation = nullptr;
            bool negate = false;
            switch (rmw->getOperation()) {
                case AtomicRMWInst::Add: operation = "atomicAdd"; break;
                case AtomicRMWInst::Sub:
                    operation = "atomicAdd";
                    negate = true;
                    break;
                case AtomicRMWInst::And: operation = "atomicAnd"; break;
                case AtomicRMWInst::Or: operation = "atomicOr"; break;
                case AtomicRMWInst::Xor: operation = "atomicXor"; break;
                case AtomicRMWInst::Min: operation = "atomicMin"; break;
                case AtomicRMWInst::Max: operation = "atomicMax"; break;
                case AtomicRMWInst::UMin: operation = "atomicMin"; break;
                case AtomicRMWInst::UMax: operation = "atomicMax"; break;
                case AtomicRMWInst::Xchg: operation = "atomicExchange"; break;
                default: break;
            }
            if (operation == nullptr) {
                fail("unsupported atomicrmw operation");
                return false;
            }
            Type *element = rmw->getValOperand()->getType();
            if (!element->isIntegerTy()) {
                fail("float atomicrmw");
                return false;
            }
            const std::string value = expr(rmw->getValOperand());
            const std::string operand =
                negate ? "(-" + value + ")" : value;
            line(tmp + " = " + operation + "(" + ref_type(element) + "(" +
                 expr(rmw->getPointerOperand()) + ").v0, " + operand + ");");
            return true;
        }
        if (isa<AtomicCmpXchgInst>(&instruction)) {
            fail("unsupported instruction: cmpxchg");
            return false;
        }
        if (auto *sw = dyn_cast<SwitchInst>(&instruction)) {
            line("switch (" + expr(sw->getCondition()) + ") {");
            for (const auto &handle : sw->cases()) {
                const int64_t label = handle.getCaseValue()->getSExtValue();
                body_ += "            case " + std::to_string(label) + ": {\n";
                body_ += edge_body(&block, handle.getCaseSuccessor());
                body_ += "            break; }\n";
            }
            body_ += "            default: {\n";
            body_ += edge_body(&block, sw->getDefaultDest());
            body_ += "            break; }\n";
            line("}");
            return true;
        }
        if (auto *extract = dyn_cast<ExtractValueInst>(&instruction)) {
            std::string access;
            Type *cur = extract->getAggregateOperand()->getType();
            for (unsigned index : extract->getIndices()) {
                auto *st = dyn_cast<StructType>(cur);
                if (st == nullptr) {
                    fail("extractvalue from a non-struct");
                    return false;
                }
                access += "." + field_name(st, index);
                cur = st->getElementType(index);
            }
            line(tmp + " = " +
                 expr(extract->getAggregateOperand()) + access + ";");
            return true;
        }
        if (auto *insert = dyn_cast<InsertValueInst>(&instruction)) {
            std::string access;
            Type *cur = insert->getAggregateOperand()->getType();
            for (unsigned index : insert->getIndices()) {
                auto *st = dyn_cast<StructType>(cur);
                if (st == nullptr) {
                    fail("insertvalue into a non-struct");
                    return false;
                }
                access += "." + field_name(st, index);
                cur = st->getElementType(index);
            }
            line(tmp + " = " +
                 expr(insert->getAggregateOperand()) + ";");
            line("    " + tmp + access + " = " + expr(insert->getInsertedValueOperand()) +
                 ";");
            return true;
        }
        if (auto *fneg = dyn_cast<UnaryOperator>(&instruction)) {
            if (fneg->getOpcode() != Instruction::FNeg) {
                fail("unsupported unary operator");
                return false;
            }
            line("    " + type_name(fneg->getType()) + " " + tmp + " = -(" +
                 expr(fneg->getOperand(0)) + ");");
            return true;
        }
        if (isa<UnreachableInst>(&instruction)) {
            line("    _pc = -1;");
            return true;
        }
        fail("unsupported instruction: " +
             std::string(instruction.getOpcodeName()));
        return false;
    }

    void emit_helpers(std::ostringstream &out) {
        if (helpers_.count("sample2darray")) {
            out << R"(
vec4 air_sample_2d_array(uint64_t tex, uint64_t smp, vec2 coord,
                         int slice, float lod, float bias) {
    // v0 contract: the low 16 bits of a Metal resource handle carry the
    // descriptor index the host bound. A real driver would resolve the
    // handle through the resource descriptor heap instead.
    int ti = int(tex & 0xffffu);
    int si = int(smp & 0xffffu);
    return texture(sampler2DArray(g_tex_2darray[nonuniformEXT(ti)],
                                  g_samplers[nonuniformEXT(si)]),
                   vec3(coord, float(slice)));
}
)";
        }
        if (helpers_.count("sample3d")) {
            out << R"(
vec4 air_sample_3d(uint64_t tex, uint64_t smp, vec3 coord, float lod,
                   float bias) {
    int ti = int(tex & 0xffffu);
    int si = int(smp & 0xffffu);
    return texture(sampler3D(g_tex_3d[nonuniformEXT(ti)],
                             g_samplers[nonuniformEXT(si)]), coord);
}
)";
        }
        if (helpers_.count("gather2darray")) {
            out << R"(
vec4 air_gather_2d_array(uint64_t tex, uint64_t smp, vec2 coord, int slice) {
    int ti = int(tex & 0xffffu);
    int si = int(smp & 0xffffu);
    return ivec4(textureGather(sampler2DArray(g_tex_2darray[nonuniformEXT(ti)],
                                              g_samplers[nonuniformEXT(si)]),
                               vec3(coord, float(slice))));
}
)";
        }
        if (helpers_.count("write2darray")) {
            out << R"(
void air_write_2d_array(uint64_t tex, ivec3 coord, vec4 value) {
    imageStore(g_images[nonuniformEXT(int(tex & 0xffffu))], coord, value);
}
)";
        }
        if (helpers_.count("sample2darraygrad")) {
            out << R"(
vec4 air_sample_2d_array_grad(uint64_t tex, uint64_t smp, vec2 coord,
                              int slice, vec2 ddx, vec2 ddy) {
    int ti = int(tex & 0xffffu);
    int si = int(smp & 0xffffu);
    return textureGrad(sampler2DArray(g_tex_2darray[nonuniformEXT(ti)],
                                      g_samplers[nonuniformEXT(si)]),
                       vec3(coord, float(slice)), ddx, ddy);
}
)";
        }
        if (helpers_.count("writebufferf")) {
            out << R"(
void air_write_buffer_f(uint64_t tex, int index, vec4 value) {
    imageStore(g_buffer_images_f[nonuniformEXT(int(tex & 0xffffu))], index,
               value);
}
)";
        }
        if (helpers_.count("samplecube")) {
            out << R"(
vec4 air_sample_cube_array(uint64_t tex, uint64_t smp, vec3 coord,
                           int slice) {
    int ti = int(tex & 0xffffu);
    int si = int(smp & 0xffffu);
    return texture(samplerCubeArray(g_tex_cube[nonuniformEXT(ti)],
                                    g_samplers[nonuniformEXT(si)]),
                   vec4(coord, float(slice)));
}
)";
        }
        if (helpers_.count("writebuffer")) {
            out << R"(
void air_write_buffer(uint64_t tex, int index, ivec4 value) {
    imageStore(g_buffer_images[nonuniformEXT(int(tex & 0xffffu))], index,
               value);
}
)";
        }
        if (helpers_.count("write2darray_int")) {
            out << R"(
void air_write_2d_array_int(uint64_t tex, ivec3 coord, ivec4 value) {
    imageStore(g_images_int[nonuniformEXT(int(tex & 0xffffu))], coord, value);
}
)";
        }
        if (helpers_.count("readms")) {
            out << R"(
vec4 air_read_2d_ms_array(uint64_t tex, ivec3 coord, int sample_index) {
    return imageLoad(g_images_ms[nonuniformEXT(int(tex & 0xffffu))], coord,
                     sample_index);
}
)";
        }
        if (helpers_.count("write3d")) {
            out << R"(
void air_write_3d(uint64_t tex, ivec3 coord, vec4 value) {
    imageStore(g_images3d[nonuniformEXT(int(tex & 0xffffu))], coord, value);
}
)";
        }
        if (helpers_.count("gather2darray_u")) {
            out << R"(
ivec4 air_gather_2d_array_u(uint64_t tex, uint64_t smp, vec2 coord,
                            int slice) {
    int ti = int(tex & 0xffffu);
    int si = int(smp & 0xffffu);
    return ivec4(textureGather(sampler2DArray(g_tex_2darray[nonuniformEXT(ti)],
                                              g_samplers[nonuniformEXT(si)]),
                               vec3(coord, float(slice))));
}
)";
        }
    }

    /// A valid zero value for `t`, or "" when GLSL has no such expression
    /// (arrays cannot be constructed).
    std::string default_value(Type *t) {
        if (t->isPointerTy()) return "0L";
        if (t->isIntegerTy(1)) return "false";
        if (t->isIntegerTy(64)) return "int64_t(0)";
        if (t->isIntegerTy()) return "0";
        if (t->isFloatingPointTy()) return "0.0";
        if (auto *st = dyn_cast<StructType>(t)) {
            const std::string name = type_name(t);
            if (name.empty()) return "";
            std::string out = name + "(";
            for (unsigned i = 0; i < st->getNumElements(); ++i) {
                const std::string field = default_value(st->getElementType(i));
                if (field.empty()) return "";
                if (i) out += ", ";
                out += field;
            }
            return out + ")";
        }
        if (t->isVectorTy()) {
            const std::string name = type_name(t);
            return name.empty() ? "" : name + "(0)";
        }
        if (t->isArrayTy()) return "";
        const std::string name = type_name(t);
        return name.empty() ? "" : name + "(0)";
    }

    /// Builds a vec4/int/float from a value of any shape, for the stubbed
    /// stage outputs (GLSL has no vec4(vec2)).
    std::string widen(const std::string &value, Type *type) {
        if (type->isVectorTy()) {
            const unsigned count =
                cast<VectorType>(type)->getElementCount().getFixedValue();
            if (!type->getScalarType()->isFloatingPointTy()) {
                // Keep integer vectors as vectors so the declared stage
                // output type and this expression agree.
                switch (count) {
                    case 4: return value;
                    case 3: return "ivec4(" + value + ", 1)";
                    case 2: return "ivec4(" + value + ", 0, 0)";
                    case 1: return "ivec4(" + value + ")";
                    default: return "ivec4(0)";
                }
            }
            switch (count) {
                case 4: return value;
                case 3: return "vec4(" + value + ", 1.0)";
                case 2: return "vec4(" + value + ", 0.0, 0.0)";
                case 1: return "vec4(" + value + ")";
                default: return "vec4(0.0)";
            }
        }
        if (type->isIntegerTy()) return "int(" + value + ")";
        return "float(" + value + ")";
    }

    /// Appends " = <value>" when a default exists, so declarations of types
    /// without constructors (arrays) stay legal.
    std::string initializer(Type *t) {
        const std::string value = default_value(t);
        return value.empty() ? std::string() : " = " + value;
    }

  public:
    std::string assemble();
};

bool Emitter::run(std::string &source, std::string &error) {
    // Entry arguments are the shader's inputs (Metal passes pointer arguments
    // as GPU addresses, so they are uint64_t here).
    for (unsigned i = 0; i < entry_->arg_size(); ++i) {
        names_[entry_->getArg(i)] = "_arg" + std::to_string(i);
    }
    // Blocks get dispatcher ids in layout order.
    for (const BasicBlock &block : *entry_) {
        block_ids_[&block] = static_cast<int>(blocks_.size());
        blocks_.push_back(&block);
    }
    // Declare every phi and alloca up front: GLSL wants declarations before
    // statements, and the dispatcher can jump to any block at any time.
    std::ostringstream declarations;
    for (const BasicBlock &block : *entry_) {
        for (const PHINode &phi : block.phis()) {
            const std::string name = "_phi" + std::to_string(phi_++);
            names_[&phi] = name;
            const std::string ty = type_name(phi.getType());
            if (ty.empty()) {
                error = error_.empty() ? "unsupported phi type" : error_;
                return false;
            }
            declarations << "    " << ty << " " << name
                         << initializer(phi.getType()) << ";\n";
        }
        for (const Instruction &instruction : block) {
            if (auto *alloca = dyn_cast<AllocaInst>(&instruction)) {
                const std::string ty = type_name(alloca->getAllocatedType());
                if (ty.empty()) {
                    error = error_.empty() ? "unsupported alloca type" : error_;
                    return false;
                }
                std::set<const Value *> seen;
                if (address_escapes(alloca, seen)) {
                    const uint64_t alignment =
                        std::max<uint64_t>(1, layout_.getABITypeAlign(
                                                  alloca->getAllocatedType())
                                                  .value());
                    scratch_size_ = (scratch_size_ + alignment - 1) / alignment *
                                    alignment;
                    scratch_[alloca] = scratch_size_;
                    scratch_size_ += layout_.getTypeAllocSize(
                        alloca->getAllocatedType());
                    continue;
                }
                const std::string name = "_a" + std::to_string(locals_.size());
                locals_[alloca] = name;
                declarations << "    " << ty << " " << name << ";\n";
            }
        }
    }

    // Every instruction result becomes a variable declared up front, for the
    // same reason as phis: any case can run before any other.
    for (const BasicBlock &block : *entry_) {
        for (const Instruction &instruction : block) {
            if (instruction.getType()->isVoidTy()) continue;
            if (isa<AllocaInst>(instruction) || isa<PHINode>(instruction) ||
                isa<GetElementPtrInst>(instruction) ||
                isa<DbgInfoIntrinsic>(instruction)) {
                continue;
            }
            const std::string name = "_t" + std::to_string(tmp_++);
            names_[&instruction] = name;
            const std::string ty = type_name(instruction.getType());
            if (ty.empty()) {
                error = error_.empty() ? "unsupported instruction type" : error_;
                return false;
            }
            declarations << "    " << ty << " " << name
                         << initializer(instruction.getType()) << ";\n";
        }
    }

    // The entry may call internal helpers (Air keeps some in the module, e.g.
    // irconverter_discard); those are only ever wrappers, so they are handled
    // by emit_intrinsic and do not need their own bodies emitted.
    body_.clear();
    for (const BasicBlock &block : *entry_) {
        if (!emit_block(block, block_ids_.at(&block))) {
            error = error_.empty() ? "translation failed" : error_;
            return false;
        }
    }

    const std::string result_type = type_name(entry_->getReturnType());
    if (result_type.empty()) {
        error = error_.empty() ? "unsupported return type" : error_;
        return false;
    }

    std::ostringstream out;
    out << "#version 460\n";
    out << "// Generated by air2glsl from Apple Air bitcode ("
        << stage_ << " shader).\n";
    out << "#extension GL_EXT_buffer_reference : require\n";
    out << "#extension GL_EXT_buffer_reference2 : require\n";
    out << "#extension GL_EXT_scalar_block_layout : require\n";
    out << "#extension GL_EXT_nonuniform_qualifier : require\n";
    out << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : "
           "require\n";
    out << "\n";
    // Separate texture and sampler arrays: the translation combines them
    // explicitly, because Metal passes them as independent arguments.
    out << "#extension GL_EXT_texture_buffer : require\n";
    out << "#extension GL_EXT_samplerless_texture_functions : require\n";
    out << "layout(set = 0, binding = 0) uniform texture2DArray "
           "g_tex_2darray[256];\n";
    out << "layout(set = 0, binding = 1) uniform texture3D g_tex_3d[64];\n";
    out << "layout(set = 0, binding = 2) uniform sampler g_samplers[16];\n";
    out << "layout(set = 0, binding = 3, rgba32f) uniform image2DArray "
           "g_images[64];\n";
    out << "layout(set = 0, binding = 4, rgba32f) uniform image3D "
           "g_images3d[32];\n";
    out << "layout(set = 0, binding = 5) uniform textureBuffer "
           "g_tex_buffer[64];\n";
    out << "layout(set = 0, binding = 10) uniform usamplerBuffer "
           "g_ubuffer[64];\n";
    out << "layout(set = 0, binding = 11, rgba32i) uniform iimage3D "
           "g_images3d_int[32];\n";
    out << "layout(set = 0, binding = 12) uniform sampler2DArrayShadow "
           "g_shadow_2darray[64];\n";
    out << "layout(set = 0, binding = 13, rgba32f) uniform imageBuffer "
           "g_buffer_images_f[64];\n";
    out << "layout(set = 0, binding = 14, rgba32i) uniform iimage2DMSArray "
           "g_images_ms_int[32];\n";
    // textureQueryLod cannot take a sampler built from a separate texture and
    // sampler, so this one binding is a combined image sampler.
    out << "layout(set = 0, binding = 15) uniform sampler2DArray "
           "g_tex_2darray_combined[64];\n";
    out << "layout(set = 0, binding = 6, rgba32f) uniform image2DMSArray "
           "g_images_ms[32];\n";
    out << "layout(set = 0, binding = 7, rgba32i) uniform iimage2DArray "
           "g_images_int[64];\n";
    out << "layout(set = 0, binding = 8) uniform textureCubeArray "
           "g_tex_cube[64];\n";
    out << "layout(set = 0, binding = 9, rgba32i) uniform iimageBuffer "
           "g_buffer_images[64];\n\n";

    // Metal globals become base addresses the host supplies. Vulkan forbids
    // non-opaque uniforms outside a block, so they live in one.
    // Metal globals become addresses the host supplies, and escapes of local
    // memory are placed in a host-allocated scratch buffer. Vulkan forbids
    // non-opaque uniforms outside a block, so they share one.
    const size_t global_count = std::max<size_t>(1, globals_.size());
    out << "// The host writes every Metal global's GPU address, and the base\n";
    out << "// of a scratch buffer of at least AIR_SCRATCH_BYTES here.\n";
    out << "#define AIR_SCRATCH_BYTES " << scratch_size_ << "\n";
    out << "layout(set = 1, binding = 0) uniform Globals {\n";
    out << "    uint64_t scratch;\n";
    out << "    uint64_t ga[" << global_count << "];\n";
    out << "} g_globals;\n\n";

    // Structs are collected while emitting, so this loop must tolerate
    // growth from the declarations it writes.
    for (size_t i = 0; i < struct_order_.size(); ++i) {
        auto *st = cast<StructType>(struct_order_[i]);
        out << "struct " << struct_names_[st] << " {\n";
        for (unsigned field = 0; field < st->getNumElements(); ++field) {
            out << "    " << type_name(st->getElementType(field)) << " "
                << field_name(st, field) << ";\n";
        }
        out << "};\n";
    }
    out << "\n";
    for (size_t i = 0; i < ref_order_.size(); ++i) {
        Type *type = ref_order_[i];
        // `buffer_reference_align` is an EXT_buffer_reference2 qualifier that
        // glslang does not recognise, so the default alignment is used.
        out << "layout(buffer_reference, scalar) buffer " << ref_names_[type]
            << " { " << type_name(type) << " v0; };\n";
    }
    out << "\n";

    // Helper functions come before their call sites.
    emit_helpers(out);

    // The translated body, as a callable function.
    std::string parameters;
    for (unsigned i = 0; i < entry_->arg_size(); ++i) {
        Type *type = entry_->getFunctionType()->getParamType(i);
        if (i) parameters += ", ";
        parameters += type_name(type) + " _arg" + std::to_string(i);
    }
    out << result_type << " air_entry(" << parameters << ") {\n";
    out << "    int _pc = 0;\n";
    if (!entry_->getReturnType()->isVoidTy()) {
        out << "    " << result_type << " _result"
            << initializer(entry_->getReturnType()) << ";\n";
    }
    out << declarations.str();
    out << "    for (int _guard = 0; _guard < 65536; ++_guard) {\n";
    out << "        switch (_pc) {\n";
    out << body_;
    out << "        default: _pc = -1; break;\n";
    out << "        }\n";
    out << "        if (_pc < 0) break;\n";
    out << "    }\n";
    if (entry_->getReturnType()->isVoidTy()) {
        out << "    return;\n";
    } else {
        out << "    return _result;\n";
    }
    out << "}\n\n";

    // v0 entry point: inputs are stubbed, outputs are written to locations.
    std::string arguments;
    for (unsigned i = 0; i < entry_->arg_size(); ++i) {
        if (i) arguments += ", ";
        const std::string value =
            default_value(entry_->getFunctionType()->getParamType(i));
        arguments += value.empty() ? "0" : value;
    }
    if (stage_ == "kernel") {
        out << "layout(local_size_x = 64) in;\n";
        out << "void main() {\n";
        if (entry_->getReturnType()->isVoidTy()) {
            out << "    air_entry(" << arguments << ");\n";
        } else {
            out << "    " << result_type << " _r = air_entry(" << arguments
                << ");\n";
            out << "    (void)_r;\n";
        }
        out << "}\n";
    } else {
        auto *st = dyn_cast<StructType>(entry_->getReturnType());
        const unsigned outputs = st ? st->getNumElements() : 1;
        for (unsigned i = 0; i < outputs; ++i) {
            const Type *type = st ? st->getElementType(i) : entry_->getReturnType();
            const bool integer = type->getScalarType()->isIntegerTy() &&
                                 !type->getScalarType()->isIntegerTy(1);
            const std::string ty =
                type->isVectorTy() ? (integer ? "ivec4" : "vec4")
                                   : (integer ? "int" : "float");
            if (stage_ == "vertex") {
                out << "layout(location = " << i << ") out " << ty << " _v"
                    << i << ";\n";
            } else {
                out << "layout(location = " << i << ") out " << ty << " _o"
                    << i << ";\n";
            }
        }
        out << "\nvoid main() {\n";
        out << "    // v0: entry inputs are stubbed; wiring up gl_FragCoord,\n";
        out << "    // vertex attributes and varyings is the next step.\n";
        if (st) {
            out << "    " << result_type << " _r = air_entry(" << arguments
                << ");\n";
            for (unsigned i = 0; i < outputs; ++i) {
                const Type *type = st->getElementType(i);
                const std::string field = "_r." + field_name(st, i);
                Type *field_type = const_cast<Type *>(type);
                if (stage_ == "vertex" && i == 0) {
                    out << "    gl_Position = " << widen(field, field_type)
                        << ";\n";
                } else {
                    out << "    _"
                        << (stage_ == "vertex" ? "v" : "o") << i << " = "
                        << widen(field, field_type) << ";\n";
                }
            }
        } else {
            out << "    air_entry(" << arguments << ");\n";
        }
        out << "}\n";
    }

    source = out.str();
    if (!unsupported_.empty()) {
        error = "unsupported: " + *unsupported_.begin();
        return false;
    }
    return true;
}

}  // namespace

std::string find_stage(const Module &module, Function *&entry) {
    return find_entry(module, entry);
}

Translation translate(const Module &module) {
    Translation result;
    Function *entry = nullptr;
    result.stage = find_entry(module, entry);
    if (entry == nullptr) {
        result.error = "no Air entry point (!air.vertex/fragment/kernel)";
        return result;
    }
    result.entry_name = entry->getName().str();
    Emitter emitter(module, entry, result.stage);
    std::string error;
    if (!emitter.run(result.source, error)) {
        result.error = error.empty() ? "translation failed" : error;
        result.source.clear();
        return result;
    }
    result.ok = true;
    return result;
}

}  // namespace air2glsl