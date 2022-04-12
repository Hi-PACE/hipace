#! /usr/bin/env python3

# Copyright 2022
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
# License: BSD-3-Clause-LBNL

# code generator for nth order taylor expansion of the green's function used in OpenBoundary.H
order = 18

import numpy as np
import sympy as sp
from sympy.printing import cxx
from sympy.printing import cxxcode
from sympy.printing import ccode
from typing import Any, Dict, Set, Tuple
from functools import wraps
from sympy.core import Add, Expr, Mul, Pow, S, sympify, Float
from sympy.core.basic import Basic
from sympy.core.compatibility import default_sort_key
from sympy.core.function import Lambda
from sympy.core.mul import _keep_coeff
from sympy.core.symbol import Symbol
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence
from sympy.codegen.ast import (
    Assignment, Pointer, Variable, Declaration, Type,
    real, complex_, integer, bool_, float32, float64, float80,
    complex64, complex128, intc, value_const, pointer_const,
    int8, int16, int32, int64, uint8, uint16, uint32, uint64, untyped,
    none
)

from sympy import init_printing
from sympy.codegen.ast import real
init_printing()

v_xfnorm = sp.Symbol('xf')
v_yfnorm = sp.Symbol('yf')

v_xsnorm = sp.Symbol('xs')
v_ysnorm = sp.Symbol('ys')

v_cx = sp.Symbol('x')
v_cy = sp.Symbol('y')

v_sval = sp.Symbol('s_v')
v_radius_2 = sp.Symbol('radius_2')

class MyCxxPrinter(sp.printing.cxx.CXX11CodePrinter):
    type_literal_suffixes = {
        float32: 'F',
        float64: '_rt',
        float80: 'L'
    }

    def _print_Pow(self, expr):
        b, e = expr.as_base_exp()
        return "pow<"+self._print(e)+">("+self._print(b)+")"

def printcxxpow(expr):
    return MyCxxPrinter().doprint(expr)


print("Generating Code up to", order, "order (exponential time complexity)")

# greens function
f_exact = sp.log((v_xfnorm-v_xsnorm)**2 + (v_yfnorm-v_ysnorm)**2)
print("Green's Function:",f_exact,'\n')

#https://stackoverflow.com/questions/22857162/multivariate-taylor-approximation-in-sympy
def Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree):
    """
    Mathematical formulation reference:
    https://math.libretexts.org/Bookshelves/Calculus/Supplemental_Modules_(Calculus)/Multivariable_Calculus/3%3A_Topics_in_Partial_Derivatives/Taylor__Polynomials_of_Functions_of_Two_Variables
    :param function_expression: Sympy expression of the function
    :param variable_list: list. All variables to be approximated (to be "Taylorized")
    :param evaluation_point: list. Coordinates, where the function will be expressed
    :param degree: int. Total degree of the Taylor polynomial
    :return: Returns a Sympy expression of the Taylor series up to a given degree, of a given multivariate expression, approximated as a multivariate polynomial evaluated at the evaluation_point
    """
    from sympy import factorial, Matrix, prod
    import itertools

    n_var = len(variable_list)
    point_coordinates = [(i, j) for i, j in (zip(variable_list, evaluation_point))]  # list of tuples with variables and their evaluation_point coordinates, to later perform substitution

    deriv_orders = list(itertools.product(range(degree + 1), repeat=n_var))  # list with exponentials of the partial derivatives
    deriv_orders = [deriv_orders[i] for i in range(len(deriv_orders)) if sum(deriv_orders[i]) <= degree]  # Discarding some higher-order terms
    n_terms = len(deriv_orders)
    deriv_orders_as_input = [list(sum(list(zip(variable_list, deriv_orders[i])), ())) for i in range(n_terms)]  # Individual degree of each partial derivative, of each term

    polynomial = 0
    for i in range(n_terms):
        partial_derivatives_at_point = function_expression.diff(*deriv_orders_as_input[i]).subs(point_coordinates)  # e.g. df/(dx*dy**2)
        denominator = prod([factorial(j) for j in deriv_orders[i]])  # e.g. (1! * 2!)
        distances_powered = prod([(Matrix(variable_list) - Matrix(evaluation_point))[j] ** deriv_orders[i][j] for j in range(n_var)])  # e.g. (x-x0)*(y-y0)**2
        polynomial += partial_derivatives_at_point / denominator * distances_powered
    return polynomial


print("Calculating Taylor expansion...")
f_approx = Taylor_polynomial_sympy(f_exact, (v_xsnorm,v_ysnorm), (0,0), order)



def get_order(fn):
    seper = sp.separatevars(fn,(v_xsnorm,v_ysnorm),dict=True)
    res = (seper[v_xsnorm] * seper[v_ysnorm]).subs(v_xsnorm,2).subs(v_ysnorm,2.0001)
    return res

print("Sorting...")
f_approx_sum = list(f_approx.args)
f_approx_sum.sort(key=get_order)

f_approx_all = [sp.separatevars(fn,(v_xsnorm,v_ysnorm),dict=True) for fn in f_approx_sum]

print("Simplifying...")
for i in range(len(f_approx_all)):
    f_approx_all[i][v_xsnorm] *= f_approx_all[i][v_ysnorm]
    f_approx_all[i][v_xsnorm] = sp.simplify(f_approx_all[i][v_xsnorm])
    f_approx_all[i]["coeff"] = sp.simplify(f_approx_all[i]["coeff"])

print("Reducing...")
for i in range(len(f_approx_all)):
    j = i
    while j<len(f_approx_all):
        if i!=j and i<len(f_approx_all) and j<len(f_approx_all):
            c = f_approx_all[j]["coeff"] / f_approx_all[i]["coeff"]
            if c.is_constant():
                c = sp.simplify(c)
                if np.abs(c)>1:
                    f_approx_all[i][v_xsnorm] = f_approx_all[i][v_xsnorm] + c*f_approx_all[j][v_xsnorm]
                    f_approx_all.remove(f_approx_all[j])
                else:
                    f_approx_all[i][v_xsnorm] = f_approx_all[i][v_xsnorm] / c + f_approx_all[j][v_xsnorm]
                    f_approx_all[i]["coeff"] *= c
                    f_approx_all.remove(f_approx_all[j])
            else:
                j+=1
        else:
            j+=1


print("Simplifying...\n")
for i in range(len(f_approx_all)):
    f_approx_all[i][v_xsnorm] = sp.simplify(f_approx_all[i][v_xsnorm])
    f_approx_all[i]["coeff"] = sp.simplify(f_approx_all[i]["coeff"])

i=0
for seper in f_approx_all:
    print("    "+printcxxpow( seper[v_xsnorm] \
        .subs( v_xsnorm, v_cx).subs( v_ysnorm, v_cy).expand()*v_sval )+",")
    i+=1

print('\n')

i=0
for seper in f_approx_all:
    print("    + amrex::get<"+str(i)+">(m_c) * ("+printcxxpow(seper["coeff"] \
        .subs(v_xfnorm, v_cx/(v_cx*v_cx+v_cy*v_cy)).subs(v_yfnorm, v_cy/(v_cx*v_cx+v_cy*v_cy)) \
        .simplify().subs(sp.log(1/(v_cx*v_cx + v_cy*v_cy)), sp.log(v_radius_2)).expand())+")")
    i+=1
