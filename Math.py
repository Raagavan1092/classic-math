from itertools import combinations, islice
from rop import read_only_properties
from copy import deepcopy
from functools import reduce
from math import degrees, acos
from typing import overload, List, Union, Tuple
import numpy as np


@read_only_properties("d_powermap")
class Term(object):
    """A monomial term with superscript for powers"""

    # d_powermap:ClassVar[dict]={8304: 0, 185: 1, 178: 2, 179: 3, 8308: 4, 8309: 5, 8310: 6, 8311: 7, 8312: 8, 8313: 9}
    # Above can be used so that d_powermap cant be changed as a Class variable, but can be done for each instance.
    def __init__(self, string: str = "") -> None:
        self.d_powermap = {8304: 0, 185: 1, 178: 2, 179: 3, 8308: 4, 8309: 5, 8310: 6, 8311: 7, 8312: 8, 8313: 9}
        if string[0] not in ["+", "-"] and len(string) != 0:
            string = "+" + string
        self.s_varexp: str = string
        self.num = ""
        self.var: str = ""
        self.d_exps: dict = {}
        self.d_varvals: dict = {}

        for i in range(len(string)):
            if ord(string[i]) in [j for j in range(48, 58)] + [43, 45]:
                self.num += string[i]
            elif ord(string[i]) in self.d_powermap.keys():
                if ord(string[i - 1]) in self.d_powermap.keys():
                    self.d_exps[list(self.d_exps.keys())[-1]] = str(self.d_exps[list(self.d_exps.keys())[-1]])
                    self.d_exps[list(self.d_exps.keys())[-1]] += str(self.d_powermap[ord(string[i])])
                    self.d_exps[list(self.d_exps.keys())[-1]] = int(self.d_exps[list(self.d_exps.keys())[-1]])
                else:
                    if string[i - 1] in list(self.d_exps.keys()):
                        self.d_exps[string[i - 1]] += self.d_powermap[ord(string[i])]
                    else:
                        self.d_exps[string[i - 1]] = self.d_powermap[ord(string[i])]
            else:
                self.var += string[i]
                try:
                    if i == len(string) - 1 or ord(string[i + 1]) in [j for j in range(65, 123)]:
                        if string[i] in list(self.d_exps.keys()):
                            self.d_exps[string[i]] += 1
                        else:
                            self.d_exps[string[i]] = 1
                except IndexError:
                    pass

        self.s_varexp = self.tempvarexp = self.s_varexp.replace(self.num, "")
        if len(self.num) == 1 and (self.num == "-" or self.num == "+"):
            self.num += "1"
        self.num: int = int(self.num)
        self.tempnum: int = self.num

        if self.var:
            for i in range(len(self.var)):
                self.d_varvals[self.var[i]] = None

        self.degree: int = 0
        for i in self.d_exps:
            self.degree += self.d_exps[i]

        self.simplify()

    def substitute(self, **kwargs: dict) -> None:
        if self.var:
            for i in kwargs:
                self.d_varvals[i] = kwargs[i]
                self.tempnum *= kwargs[i] ** self.d_exps[i]
                if self.d_exps[i] > 1:
                    self.tempvarexp = self.s_varexp.replace(i + str(self.d_exps[i]).translate(superscript), "")
                else:
                    self.tempvarexp = self.tempvarexp.replace(i, "")

    def reset(self) -> None:
        self.tempvarexp = self.s_varexp
        self.tempnum = self.num
        for i in self.var:
            self.d_varvals[i] = None

    def simplify(self) -> None:
        if len(self.var) != len(self.d_exps):
            self.var = self.s_varexp = ""
            for i in self.d_exps:
                self.var += i
                if self.d_exps[i] == 1:
                    self.s_varexp += i
                else:
                    self.s_varexp += i + str(self.d_exps[i]).translate(superscript)

    def __add__(self, other) -> Union["Term", "Polynomial"]:
        if isinstance(other, Term):
            if self.d_exps == other.d_exps:
                return Term(str(self.num + other.num) + self.s_varexp)
            else:
                return Polynomial(str(self) + str(other))
        elif isinstance(other, (int, float)):
            return Polynomial(f"{self}+{other}")

    def __mul__(self, other) -> Union["Term", "Polynomial"]:
        if isinstance(other, (float, int)):
            return Term(str(self.num * other) + self.s_varexp)
        elif isinstance(other, Term):
            return Term(str(self.num * other.num) + self.s_varexp + other.s_varexp)
        elif isinstance(other, Polynomial):
            exprnew = ""
            for i in range(len(other.l_terms)):
                other.l_terms[i] *= self
                exprnew += str(other.l_terms[i])
            return Polynomial(exprnew)

    def __pow__(self, power: int, modulo=None) -> "Term":
        if isinstance(power, int):
            strpart = ""
            for i in self.var:
                if power:
                    expostr = str(self.d_exps[i] * power).translate(superscript)
                    if self.d_exps[i] * power == 1:
                        strpart += i
                    else:
                        strpart += i + expostr
            numpart = self.num ** power
            return Term(str(numpart) + strpart)
        else:
            raise TypeError("Cannot exponentiate term with any other type than int")

    def __gt__(self, other) -> bool:
        if isinstance(other, Term):
            if self.degree > other.degree:
                return False
            elif self.degree == other.degree:
                if len(self.var) > len(other.var):
                    return True
                elif len(self.var) == len(other.var):
                    if self.var > other.var:
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return True

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self.num > 0:
            return f"+{self.num}{self.s_varexp}"
        elif self.num < 0:
            return f"{self.num}{self.s_varexp}"
        else:
            return "0"


class Polynomial:
    """An expression of more than one term"""

    @overload
    def __init__(self, expression: str):
        self.l_terms = self.expr = ""

    @overload
    def __init__(self, term_list: List["Term"]):
        pass

    def __init__(self, expression) -> None:
        self.l_terms: List["Term"] = []
        if isinstance(expression, str):
            self.expr: str = expression
            term = ""
            for i in range(len(self.expr)):
                if self.expr[i] == " ":
                    continue
                elif self.expr[i] in ["+", "-"]:
                    if term == "":
                        term += self.expr[i]
                        continue
                    self.l_terms.append(Term(term))
                    term = self.expr[i]  # cleared the previous stuff and assigned the next operator
                elif i == len(self.expr) - 1:
                    term += self.expr[i]
                    self.l_terms.append(Term(term))
                else:
                    term += self.expr[i]
        elif isinstance(expression, list):
            self.expr = ""
            for i in expression:
                if not isinstance(i, Term):
                    raise TypeError("List must contain only Term objects")
            else:
                self.l_terms = expression
                for i in self.l_terms:
                    self.expr += str(i)
        else:
            raise ValueError("Either a string or a list of Term objects must be passed")
        self.simplify()

    def simplify(self) -> None:
        self.l_terms.sort()
        arr_terms, self.l_terms = np.array(self.l_terms), []  # Arrays supports multiple deletion at same time. So used
        while len(arr_terms) != 0:
            if len(arr_terms) == 1:
                self.l_terms.append(arr_terms[0])
                break
            # term is the one to be compares, isMatched tells whether like terms are found, l_indices are match-indices
            term, isMatched, l_indices = arr_terms[0], False, [0]
            for i in range(1, len(arr_terms)):
                if term.d_exps == arr_terms[i].d_exps:
                    isMatched = True
                    l_indices.append(i)
            if not isMatched:
                self.l_terms.append(term)
            else:
                self.l_terms.append(reduce(lambda x, y: x + y, [arr_terms[j] for j in l_indices]))  # Add all matches
            arr_terms = np.delete(arr_terms, tuple(l_indices))  # To exit while loop eventually
        self.expr = reduce(lambda x, y: str(x) + str(y), self.l_terms)

    def __add__(self, other) -> "Polynomial":
        if isinstance(other, (Polynomial, Term)):
            return Polynomial(f"{self}{other}")
        elif isinstance(other, (int, float)):
            return Polynomial(f"{self}+{other}")

    def __mul__(self, other) -> "Polynomial":
        exprnew = ""
        if isinstance(other, (float, int)):
            for i in range(len(self.l_terms)):
                self.l_terms[i].num *= other
                exprnew += str(self.l_terms[i])
        elif isinstance(other, Term):
            for i in range(len(self.l_terms)):
                self.l_terms[i] *= other
                exprnew += str(self.l_terms[i])
        else:
            for i in range(len(self.l_terms)):
                for j in range(len(other.l_terms)):
                    exprnew += str(self.l_terms[i] * other.l_terms[j])
        return Polynomial(exprnew)

    def __pow__(self, power: int, modulo=None) -> "Polynomial":
        if isinstance(power, int):
            # l_combs is the sets of nos that are in a Pascal's triangle
            l_combs = [len(list(combinations(range(power), i))) for i in range(power + 1)]
            l_new = []
            if len(self.l_terms) == 1:
                l_new.append(self.l_terms[0] ** power)
            elif len(self.l_terms) == 2:
                for i in range(power + 1):
                    newterm = (self.l_terms[0] ** (power - i)) * (self.l_terms[1] ** i)
                    newterm.num *= l_combs[i]
                    l_new.append(newterm)
            elif len(self.l_terms) > 2:
                l1, l2 = [], []
                for i in range(len(self.l_terms)):
                    if len(self.l_terms) % 2 == 0:
                        l1, l2 = self.l_terms[:len(self.l_terms) // 2], self.l_terms[len(self.l_terms) // 2:]
                    else:
                        l1, l2 = self.l_terms[:len(self.l_terms) // 2 + 1], self.l_terms[len(self.l_terms) // 2 + 1:]
                for i in range(power + 1):
                    newpol = (Polynomial(l1) ** (power - i)) * (Polynomial(l2) ** i)
                    newpol *= l_combs[i]
                    l_new.extend(newpol.l_terms)
            return Polynomial(l_new)
        else:
            raise TypeError("Cannot exponentiate with any other type than int")

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return self.expr


class Matrix:
    """Matrix object with an elementlist whose elements are matrix's row-wise elements"""

    def __init__(self, elementlist: List[Union[float, int, "Term", "Polynomial", "Vector"]],
                 nrows: int, ncolumns: int) -> None:
        if nrows*ncolumns == len(elementlist):
            self.rows: int = nrows
            self.columns: int = ncolumns
            self.elemlist: list = elementlist

            # Separating the element list into different sublists to signify rows
            lrow_iter = iter(elementlist)
            self.lrows: List[list] = [list(islice(lrow_iter, e)) for e in self.rows * [self.columns]]

            # Preparing an element list column wise to do the same as above
            self.lcolumns: List[list] = [k[i] for i in range(self.columns) for k in self.lrows]
            lcol_iter = iter(self.lcolumns)
            self.lcolumns = [list(islice(lcol_iter, e)) for e in self.columns * [self.rows]]

    def __add__(self, other) -> "Matrix":
        if self.rows * self.columns == other.rows * other.columns:
            m_added = Matrix(self.rows * self.columns * [0], self.rows, self.columns)
            m_added.elemlist.clear()
            for i in range(self.rows):
                for j in range(self.columns):
                    m_added.lrows[i][j] = (self.lrows[i][j] + other.lrows[i][j])
                    m_added.elemlist.append((self.lrows[i][j] + other.lrows[i][j]))

            for i in range(self.columns):
                for j in range(self.rows):
                    m_added.lcolumns[i][j] = (self.lcolumns[i][j] + other.lcolumns[i][j])
            return m_added
        else:
            return []

    def __sub__(self, other) -> "Matrix":
        if self.rows * self.columns == other.rows * other.columns:
            m_subtracted = Matrix(self.rows * self.columns * [0], self.rows, self.columns)
            m_subtracted.elemlist.clear()
            for i in range(self.rows):
                for j in range(self.columns):
                    m_subtracted.lrows[i][j] = (self.lrows[i][j] - other.lrows[i][j])
                    m_subtracted.elemlist.append(self.lrows[i][j] - other.lrows[i][j])
            for i in range(self.columns):
                for j in range(self.rows):
                    m_subtracted.lcolumns[i][j] = (self.lcolumns[i][j] - other.lcolumns[i][j])

            return m_subtracted
        else:
            return []

    def __truediv__(self, other) -> "Matrix":
        if self.rows * self.columns == other.rows * other.columns:
            m_divided = Matrix(self.rows * self.columns * [0], self.rows, self.columns)
            m_divided.elemlist.clear()

            for i in range(self.rows):
                for j in range(self.columns):
                    try:
                        # the condition is to avoid results such as 10 / 5 = 2.0 to give just 2
                        if not str(round((self.lrows[i][j] / other.lrows[i][j]), 2)).split(".")[1] == "0":
                            m_divided.lrows[i][j] = round((self.lrows[i][j] / other.lrows[i][j]), 2)
                            m_divided.elemlist.append(round((self.lrows[i][j] / other.lrows[i][j]), 2))
                        else:
                            m_divided.lrows[i][j] = int(round((self.lrows[i][j] / other.lrows[i][j]), 2))
                            m_divided.elemlist.append(int(round((self.lrows[i][j] / other.lrows[i][j]), 2)))
                    except ZeroDivisionError:
                        return []
            for i in range(self.columns):
                for j in range(self.rows):
                    if not str((round((self.lrows[i][j] / other.lrows[i][j]), 2))).split(".")[1] == "0":
                        m_divided.lcolumns[i][j] = (round((self.lrows[i][j] / other.lrows[i][j]), 2))
                    else:
                        m_divided.lcolumns[i][j] = int(round((self.lrows[i][j] / other.lrows[i][j]), 2))
            return m_divided
        else:
            return []

    def __mul__(self, other) -> "Matrix":
        if self.columns == other.rows:
            m_multiplied = Matrix(self.rows * other.columns * [0], self.rows, other.columns)
            m_multiplied.elemlist.clear()
            for i in range(self.rows):
                for j in range(other.columns):
                    sumforelem = 0
                    for k in range(self.columns):
                        sumforelem += self.lrows[i][k] * other.lcolumns[j][k]
                    m_multiplied.elemlist.append(sumforelem)
                    m_multiplied.lrows[i][j] = sumforelem

            m_multiplied.lcolumns.clear()    # Different from above as diplacing list items will take more lines
            for i in range(other.columns):
                for k in m_multiplied.lrows:
                    m_multiplied.lcolumns.append(k[i])

            lcol_iter = iter(m_multiplied.lcolumns)
            m_multiplied.lcolumns = [list(islice(lcol_iter, e)) for e in other.columns * [self.rows]]

            return m_multiplied
        else:
            return []

    def __str__(self) -> str:  # todo: Look if np arrays can be used and devise way to express here
        return f"Matrix of order {self.rows}x{self.columns}"
# todo: Do Fraction class


class Determinant:
    """Determinant object with element list of row-wise elements of the matrix"""

    def __init__(self, elementlist: List[Union[int, float, "Term", "Polynomial", "Vector"]], size: int) -> None:
        if size ** 2 == len(elementlist):
            self.elemlist: list = elementlist
            self.size: int = size
            self.rows: int = size
            self.columns: int = size

            lrow_iter = iter(self.elemlist)
            self.lrows: List[list] = [list(islice(lrow_iter, e)) for e in self.size * [self.size]]

            self.lcolumns: List[list] = [k[i] for i in range(self.size) for k in self.lrows]
            lcol_iter = iter(self.lcolumns)
            self.lcolumns = [list(islice(lcol_iter, e)) for e in self.size * [self.size]]

            if self.size == 1:
                self.detvalue = self.elemlist[0]
            elif self.size >= 2:
                l_subvals = []
                for i in range(self.size):
                    col, cof = self.lcolumns[i][0], self.cofactor(1, i + 1)
                    # Laplace's expansion of the first row is done here
                    if isinstance(col, Vector) and isinstance(cof, Vector):
                        l_subvals.append(Vector.DotProduct(col, cof))
                    elif isinstance(col, (float, int)) and isinstance(cof, Vector):
                        l_subvals.append(cof * col)
                    else:
                        l_subvals.append(col * cof)
                self.detvalue = reduce((lambda d1, d2: d1 + d2), l_subvals)
        else:
            raise ValueError("Your entry is invalid")

    def minor(self, elemrow: int, elemcol: int) -> Union[int, float, "Term", "Polynomial", "Vector"]:
        if self.size == 1 and (elemrow == 1 and elemcol == 1):
            return self.elemlist[0]

        elif self.size >= 2 and (elemrow in [x+1 for x in range(self.size)]
                                 and elemcol in [x+1 for x in range(self.size)]):
            # Very important part of the class
            rowslist = deepcopy(self.lrows)    # So that original list has no changes done to it
            rowslist.pop(elemrow - 1)
            for i in range(self.size - 1):
                rowslist[i].pop(elemcol - 1)
            lelems = [i[j] for i in rowslist for j in range(self.size - 1)]
            # The necassary elements removed, lelems is row-wise list of the smaller determinant
            det = Determinant(lelems, self.size - 1)    # Recursion of the same class
            return det.detvalue

    def cofactor(self, elemrow: int, elemcol: int) -> Union[int, float, "Term", "Polynomial", "Vector"]:
        negorpos = (-1) ** (elemrow + elemcol)
        minor = self.minor(elemrow, elemcol)
        if minor is not None:
            return minor * negorpos

    def __repr__(self) -> str:
        return str(self.detvalue)

    def __str__(self) -> str:
        return f"\u0394 = {self.detvalue}"


class Vector:
    """Vector object with general attributes and methods applicable for a typical vector. Operations are supported.
    Numerical attributes are returned as long float objects. Please use round() for your convenience"""

    @overload
    def __init__(self, vector: "Vector"): ...

    @overload
    def __init__(self, Complex: complex): ...

    @overload
    def __init__(self, x_comp=0.0, y_comp=0.0, z_comp=0.0): ...

    def __init__(self, x_comp=0.0, y_comp=0.0, z_comp=0.0) -> None:
        if isinstance(x_comp, Vector) and not y_comp and not z_comp:
            self.x, self.y, self.z = x_comp.x, x_comp.y, x_comp.z
        elif isinstance(x_comp, complex) and not y_comp and not z_comp:
            self.x, self.y, self.z = x_comp.real, x_comp.imag, 0.0
        elif isinstance(x_comp, (int, float)) and isinstance(y_comp, (int, float)) and isinstance(z_comp, (int, float)):
            self.x, self.y, self.z = float(x_comp), float(y_comp), float(z_comp)
        else:
            raise TypeError("The type of values passed is unacceptable")

        if (self.x, self.y, self.z) == (0, 0, 0):
            self.dimension = 0
        elif not (self.x, self.y, self.z) == (0, 0, 0):
            self.dimension: int = 3
        elif not (self.x, self.y) == (0, 0) or (self.y, self.z) == (0, 0) or (self.x, self.z) == (0, 0):
            self.dimension = 2
        else:
            self.dimension = 1
        self.length: float = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
        try:
            self.dircos_l: float = self.x / self.length
            self.dircos_m: float = self.y / self.length
            self.dircos_n: float = self.z / self.length
            float()
            self.alpha: float = degrees(acos(self.dircos_l))
            self.beta: float = degrees(acos(self.dircos_m))
            self.gamma: float = degrees(acos(self.dircos_n))
        except ZeroDivisionError:
            self.dircos_l = self.dircos_m = self.dircos_n = None  # They are actually infinite

    def UnitVector(self) -> "Vector":
        if self.dircos_l is not None and self.dircos_m is not None and self.dircos_n is not None:
            return Vector(self.dircos_l, self.dircos_m, self.dircos_n)

    @classmethod
    def FromTwoPoints(cls, A: Tuple[int, int, int], B: Tuple[int, int, int]) -> "Vector":
        return Vector(B[0] - A[0], B[1] - A[1], B[2] - A[2])

    @classmethod
    def DotProduct(cls, vector1: "Vector", vector2: "Vector") -> float:
        return (vector1.x * vector2.x) + (vector1.y * vector2.y) + (vector1.z * vector2.z)

    @classmethod
    def CrossProduct(cls, vector1: "Vector", vector2: "Vector"):
        # Can use Determinant class here. But this is more simpler
        return Vector((vector1.y * vector2.z - vector1.z * vector2.y), -(vector1.x * vector2.z - vector1.z * vector2.x),
                      (vector1.x * vector2.y - vector1.y * vector2.x))

    @classmethod
    def angle(cls, vector1: "Vector", vector2: "Vector") -> float:
        return degrees(acos(cls.DotProduct(vector1, vector2) / (vector1.length * vector2.length)))

    def __add__(self, other) -> "Vector":
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other) -> "Vector":
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Vector(other * self.x, other * self.y, other * self.z)
        elif isinstance(other, Vector):
            print("Please use class methods - Vector.DotProduct() or Vector.CrossProduct()")

    def __neg__(self) -> "Vector":
        return Vector(-self.x, -self.y, -self.z)

    def __complex__(self) -> complex:
        if self.dimension <= 2:
            return complex(self.x, self.y)

    def __abs__(self) -> float:  # because we use the modulus symbol to express length of vector
        return self.length

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        x = f"+{self.x}\u00ee " if self.x > 0 else (f"{self.x}\u00ee " if self.x < 0 else "")
        y = f"+{self.y}\u0135 " if self.y > 0 else (f"{self.y}\u0135 " if self.y < 0 else "")
        z = f"+{self.z}\u006b\u0302" if self.z > 0 else (f"{self.z}\u006b\u0302" if self.z < 0 else "")
        if x == "" and y == "" and z == "":
            return f"Vector(0)"
        else:
            return f"Vector({x}{y}{z})"


superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
if __name__ == "__main__":
    pass
