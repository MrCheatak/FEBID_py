# encoding: utf-8
# module numexpr.interpreter
# from /Users/sandrik1742/.conda/envs/FEBID/lib/python3.8/site-packages/numexpr/interpreter.cpython-38-darwin.so
# by generator 1.147
# no doc
# no imports

# Variables with simple values

allaxes = 255

maxdims = 32

MAX_THREADS = 64

use_vml = False

__BLOCK_SIZE1__ = 1024

# functions

def _get_num_threads(*args, **kwargs): # real signature unknown
    """ Gets the maximum number of threads currently in use for operations. """
    pass

def _set_num_threads(*args, **kwargs): # real signature unknown
    """ Suggests a maximum number of threads to be used in operations. """
    pass

# classes

class NumExpr(object):
    """ NumExpr objects """
    def run(self, *args, **kwargs): # real signature unknown
        pass

    def __call__(self, *args, **kwargs): # real signature unknown
        """ Call self as a function. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    constants = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    constsig = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    fullsig = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    input_names = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    program = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    signature = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    tempsig = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default



# variables with complex values

funccodes = {
    b'absolute_cc': 18,
    b'absolute_dd': 18,
    b'absolute_ff': 18,
    b'arccos_cc': 5,
    b'arccos_dd': 5,
    b'arccos_ff': 5,
    b'arccosh_cc': 11,
    b'arccosh_dd': 11,
    b'arccosh_ff': 11,
    b'arcsin_cc': 4,
    b'arcsin_dd': 4,
    b'arcsin_ff': 4,
    b'arcsinh_cc': 10,
    b'arcsinh_dd': 10,
    b'arcsinh_ff': 10,
    b'arctan2_ddd': 1,
    b'arctan2_fff': 1,
    b'arctan_cc': 6,
    b'arctan_dd': 6,
    b'arctan_ff': 6,
    b'arctanh_cc': 12,
    b'arctanh_dd': 12,
    b'arctanh_ff': 12,
    b'ceil_dd': 20,
    b'ceil_ff': 20,
    b'conjugate_cc': 19,
    b'conjugate_dd': 19,
    b'conjugate_ff': 19,
    b'cos_cc': 2,
    b'cos_dd': 2,
    b'cos_ff': 2,
    b'cosh_cc': 8,
    b'cosh_dd': 8,
    b'cosh_ff': 8,
    b'exp_cc': 16,
    b'exp_dd': 16,
    b'exp_ff': 16,
    b'expm1_cc': 17,
    b'expm1_dd': 17,
    b'expm1_ff': 17,
    b'floor_dd': 21,
    b'floor_ff': 21,
    b'fmod_ddd': 0,
    b'fmod_fff': 0,
    b'log10_cc': 15,
    b'log10_dd': 15,
    b'log10_ff': 15,
    b'log1p_cc': 14,
    b'log1p_dd': 14,
    b'log1p_ff': 14,
    b'log_cc': 13,
    b'log_dd': 13,
    b'log_ff': 13,
    b'pow_ccc': 0,
    b'sin_cc': 1,
    b'sin_dd': 1,
    b'sin_ff': 1,
    b'sinh_cc': 7,
    b'sinh_dd': 7,
    b'sinh_ff': 7,
    b'sqrt_cc': 0,
    b'sqrt_dd': 0,
    b'sqrt_ff': 0,
    b'tan_cc': 3,
    b'tan_dd': 3,
    b'tan_ff': 3,
    b'tanh_cc': 9,
    b'tanh_dd': 9,
    b'tanh_ff': 9,
}

opcodes = {
    b'add_ccc': 93,
    b'add_ddd': 74,
    b'add_fff': 58,
    b'add_iii': 31,
    b'add_lll': 44,
    b'and_bbb': 3,
    b'cast_cd': 89,
    b'cast_cf': 88,
    b'cast_ci': 86,
    b'cast_cl': 87,
    b'cast_df': 70,
    b'cast_di': 68,
    b'cast_dl': 69,
    b'cast_fi': 53,
    b'cast_fl': 54,
    b'cast_ib': 27,
    b'cast_li': 40,
    b'complex_cdd': 102,
    b'contains_bss': 105,
    b'copy_bb': 1,
    b'copy_cc': 91,
    b'copy_dd': 71,
    b'copy_ff': 55,
    b'copy_ii': 28,
    b'copy_ll': 41,
    b'copy_ss': 103,
    b'div_ccc': 96,
    b'div_ddd': 77,
    b'div_fff': 61,
    b'div_iii': 34,
    b'div_lll': 47,
    b'eq_bbb': 5,
    b'eq_bcc': 84,
    b'eq_bdd': 21,
    b'eq_bff': 17,
    b'eq_bii': 9,
    b'eq_bll': 13,
    b'eq_bss': 25,
    b'func_cccn': 99,
    b'func_ccn': 98,
    b'func_dddn': 83,
    b'func_ddn': 82,
    b'func_fffn': 67,
    b'func_ffn': 66,
    b'ge_bdd': 20,
    b'ge_bff': 16,
    b'ge_bii': 8,
    b'ge_bll': 12,
    b'ge_bss': 24,
    b'gt_bdd': 19,
    b'gt_bff': 15,
    b'gt_bii': 7,
    b'gt_bll': 11,
    b'gt_bss': 23,
    b'imag_dc': 101,
    b'invert_bb': 2,
    b'lshift_iii': 37,
    b'lshift_lll': 50,
    b'max_ddn': 127,
    b'max_ffn': 126,
    b'max_iin': 124,
    b'max_lln': 125,
    b'min_ddn': 122,
    b'min_ffn': 121,
    b'min_iin': 119,
    b'min_lln': 120,
    b'mod_ddd': 79,
    b'mod_fff': 63,
    b'mod_iii': 36,
    b'mod_lll': 49,
    b'mul_ccc': 95,
    b'mul_ddd': 76,
    b'mul_fff': 60,
    b'mul_iii': 33,
    b'mul_lll': 46,
    b'ne_bbb': 6,
    b'ne_bcc': 85,
    b'ne_bdd': 22,
    b'ne_bff': 18,
    b'ne_bii': 10,
    b'ne_bll': 14,
    b'ne_bss': 26,
    b'neg_cc': 92,
    b'neg_dd': 73,
    b'neg_ff': 57,
    b'neg_ii': 30,
    b'neg_ll': 43,
    b'noop': 0,
    b'ones_like_cc': 90,
    b'ones_like_dd': 72,
    b'ones_like_ff': 56,
    b'ones_like_ii': 29,
    b'ones_like_ll': 42,
    b'or_bbb': 4,
    b'pow_ddd': 78,
    b'pow_fff': 62,
    b'pow_iii': 35,
    b'pow_lll': 48,
    b'prod_ccn': 117,
    b'prod_ddn': 116,
    b'prod_ffn': 115,
    b'prod_iin': 113,
    b'prod_lln': 114,
    b'real_dc': 100,
    b'rshift_iii': 38,
    b'rshift_lll': 51,
    b'sqrt_dd': 80,
    b'sqrt_ff': 64,
    b'sub_ccc': 94,
    b'sub_ddd': 75,
    b'sub_fff': 59,
    b'sub_iii': 32,
    b'sub_lll': 45,
    b'sum_ccn': 111,
    b'sum_ddn': 110,
    b'sum_ffn': 109,
    b'sum_iin': 107,
    b'sum_lln': 108,
    b'where_bbbb': 104,
    b'where_cbcc': 97,
    b'where_dbdd': 81,
    b'where_fbff': 65,
    b'where_ibii': 39,
    b'where_lbll': 52,
}

__loader__ = None # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x7fca54eb82b0>'

__spec__ = None # (!) real value is "ModuleSpec(name='numexpr.interpreter', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x7fca54eb82b0>, origin='/Users/sandrik1742/.conda/envs/FEBID/lib/python3.8/site-packages/numexpr/interpreter.cpython-38-darwin.so')"

